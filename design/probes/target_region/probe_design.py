#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Design target regions from transcript FASTA sequences and BLAST hits.

The implementation follows the published target-region filtering procedure;
the original third-party attribution is retained below.

Copyright 2021 Chenliu.Tang lab. USTC — Python port
"""

from __future__ import annotations

import argparse
import csv
import glob
import io
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from collections import OrderedDict
from typing import (Callable, Dict, Iterable, Iterator, List, Optional,
                    Sequence, Tuple)

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from oligoprop import check_hairpin_dimer, gc_and_mean_tm  # noqa: E402

__version__ = "1.0"

# --------------------------------------------------------------------------- #
# Default parameters: correspond item by item to lines 13-29 of
# startup_bassfish_blast.m.
# --------------------------------------------------------------------------- #
DEFAULTS = {
    "fanxiang": 0,          # 1 = reverse-complement probes, 0 = transcript orientation (usually 0)
    "isoform_spec": 0,      # whether to account for isoform specificity: 0 = no, 1 = yes
    "binding_length": 38,   # total binding-region length (must be even)
    "looooong": 5,          # exclude homopolymers of this length, default 5, e.g. 'GGGGG'
    "min_tm": 52.0,         # minimum acceptable Tm
    "max_tm": 70.0,         # maximum acceptable Tm
    "min_gc": 40.0,         # minimum acceptable GC percentage (%)
    "max_gc": 65.0,         # maximum acceptable GC percentage (%)
    "maxdm": 10,            # maximum dimer length of probe
    "maxstem": 8,           # maximum stem length of probe
    "mid_length": 6,        # minimum disallowed central mismatch-connection length (must be even)
    "singlehomo": 15,       # maximum one-sided transcript homology length
    "bothhomo": 26,         # maximum continuous homology of one probe to the transcript
    "gaplength": 2,         # minimum gap between probes for one gene (declared but unused in MATLAB)
    "evalue": 1000.0,       # BLAST e-value
}

# Species presets: database name + transcript FASTA filename (matching target_region_matlab).
SPECIES_PRESET = {
    "human": {"db": "hdb", "fasta": "gencode_v42_transcripts.fasta"},
    "mouse": {"db": "mdb", "fasta": "gencode.vM26.transcripts.fasta"},
}

# Resource defaults for local execution.
DEFAULT_MAX_THREADS = 3
DEFAULT_SEQ_CACHE_MB = 192
DEFAULT_MIN_FREE_GB = 2.0
DEFAULT_MEMORY_BUDGET_MB = 4096

# MATLAB read_and_clear_probe_txt.m matches reference transcripts using Header(1:44).
HEADER_KEY_LEN = 44
# MATLAB probe_nonspecific_binding_check_hamming.m masks the target transcript with 47 A's.
TARGET_MASK_SEQ = "A" * 47

_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")
# Candidate probes must contain only ACGT; discard N and all other ambiguous bases.
_ACGT_ONLY_RE = re.compile(r"^[ACGT]+$")


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# 0. Detect runtime resources (in containers, os.cpu_count() reports host CPUs;
# cgroup limits must also be checked).
# --------------------------------------------------------------------------- #
def _read_first_line(path: str) -> Optional[str]:
    try:
        with open(path, "r") as fh:
            return fh.readline().strip()
    except (OSError, IOError):
        return None


def detect_cpu_limit() -> int:
    """Return the CPU count actually available to this process.

    Check cgroup v2/v1, CPU affinity, and cpu_count in that order.
    """
    limits: List[int] = []

    line = _read_first_line("/sys/fs/cgroup/cpu.max")          # cgroup v2
    if line:
        parts = line.split()
        if len(parts) == 2 and parts[0] != "max":
            try:
                quota, period = int(parts[0]), int(parts[1])
                if quota > 0 and period > 0:
                    limits.append(max(1, quota // period))
            except ValueError:
                pass

    quota_s = _read_first_line("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")   # cgroup v1
    period_s = _read_first_line("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota_s and period_s:
        try:
            quota, period = int(quota_s), int(period_s)
            if quota > 0 and period > 0:
                limits.append(max(1, quota // period))
        except ValueError:
            pass

    if hasattr(os, "sched_getaffinity"):
        try:
            limits.append(len(os.sched_getaffinity(0)))         # type: ignore[attr-defined]
        except OSError:
            pass

    limits.append(os.cpu_count() or 1)
    return max(1, min(limits))


def detect_memory_limit_mb() -> Optional[int]:
    """Return the cgroup memory limit in MB, or None if it cannot be detected."""
    for path in ("/sys/fs/cgroup/memory.max",
                 "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        line = _read_first_line(path)
        if not line or line == "max":
            continue
        try:
            value = int(line)
        except ValueError:
            continue
        if 0 < value < (1 << 50):     # Ignore the astronomical value used for no limit.
            return value // 1048576
    return None


def peak_rss_mb() -> Optional[float]:
    """Return this process's peak resident memory in MB, or None on Windows."""
    try:
        import resource
    except ImportError:
        return None
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports ru_maxrss in bytes; Linux reports KB.
    return value / 1048576.0 if sys.platform == "darwin" else value / 1024.0


class RunLock(object):
    """Cross-platform process lock for serializing competing runs."""

    def __init__(self, path: Optional[str], timeout: float = 0.0):
        self.path = path
        self.timeout = timeout
        self._fh = None

    def _try_acquire(self) -> bool:
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except (OSError, IOError, ImportError):
            return False

    def __enter__(self) -> "RunLock":
        if not self.path:
            return self
        self._fh = open(self.path, "a+")
        deadline = time.time() + max(0.0, self.timeout)
        waited = False
        while True:
            if self._try_acquire():
                if waited:
                    log("Run lock acquired; starting")
                return self
            if time.time() >= deadline:
                self._fh.close()
                self._fh = None
                raise RuntimeError("Another run holds lock file %s; it was not released after waiting %.0f s"
                                   % (self.path, self.timeout))
            if not waited:
                log("Another run holds the lock file; waiting (up to %.0f s)..." % self.timeout)
                waited = True
            time.sleep(1.0)

    def __exit__(self, *_exc) -> None:
        if self._fh is None:
            return
        try:
            if os.name == "nt":
                import msvcrt
                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
        except (OSError, IOError, ImportError):
            pass
        self._fh.close()      # Closing the fd releases flock on POSIX.
        self._fh = None


def free_disk_bytes(path: str) -> int:
    probe = path
    while probe and not os.path.isdir(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    try:
        return shutil.disk_usage(probe or ".").free
    except OSError:
        return 1 << 62      # Treat an unavailable value as ample; do not block the workflow.


# --------------------------------------------------------------------------- #
# 1. Transcript FASTA index (lazy reads avoid loading the entire 400+ MB reference).
# --------------------------------------------------------------------------- #
class FastaIndex(object):
    """Build a per-record (header, sequence byte-span) index and read records on demand.

    The only difference from MATLAB ``fastaread`` is memory usage; ``header(i)`` /
    ``sequence(i)`` correspond to ``washed_tp(i).Header`` / ``washed_tp(i).Sequence``
    (with zero-based indices).
    """

    INDEX_VERSION = 1

    def __init__(self, path: str, use_cache: bool = True,
                 cache_mb: int = DEFAULT_SEQ_CACHE_MB):
        self.path = os.path.abspath(path)
        if not os.path.isfile(self.path):
            raise FileNotFoundError("Transcript FASTA not found: %s" % self.path)
        self.headers: List[str] = []
        self.spans: List[Tuple[int, int]] = []
        self._fh: Optional[io.BufferedReader] = None
        # Bound by bytes rather than record count: one transcript can reach 350,000 nt,
        # so a record-count limit can exhaust memory.
        self._seq_cache: "OrderedDict[int, str]" = OrderedDict()
        self._cache_bytes = 0
        self._cache_limit = max(1, int(cache_mb)) * 1048576
        self._overrides: Dict[int, str] = {}
        self._build(use_cache)
        self._key_map: Dict[str, int] = {}
        for i, header in enumerate(self.headers):
            self._key_map.setdefault(header[:HEADER_KEY_LEN], i)

    # -- Build / cache ------------------------------------------------------- #
    def _cache_path(self) -> str:
        return self.path + ".pdidx"

    def _build(self, use_cache: bool) -> None:
        stat = os.stat(self.path)
        signature = (self.INDEX_VERSION, stat.st_size, int(stat.st_mtime))
        cache_file = self._cache_path()

        if use_cache and os.path.isfile(cache_file):
            try:
                with open(cache_file, "rb") as fh:
                    blob = pickle.load(fh)
                if blob.get("signature") == signature:
                    self.headers = blob["headers"]
                    self.spans = blob["spans"]
                    log("  Reused FASTA index cache (%d transcripts)" % len(self.headers))
                    return
            except Exception:
                pass  # Rebuild a corrupt cache.

        t0 = time.time()
        headers: List[str] = []
        spans: List[Tuple[int, int]] = []
        with open(self.path, "rb") as fh:
            pos = 0
            cur_header = None
            seq_start = 0
            for line in fh:
                n = len(line)
                if line[:1] == b">":
                    if cur_header is not None:
                        headers.append(cur_header)
                        spans.append((seq_start, pos))
                    cur_header = line[1:].rstrip(b"\r\n").decode("utf-8", "replace")
                    seq_start = pos + n
                pos += n
            if cur_header is not None:
                headers.append(cur_header)
                spans.append((seq_start, pos))
        self.headers = headers
        self.spans = spans
        log("  Built FASTA index: %d transcripts in %.1f s" % (len(headers), time.time() - t0))

        if use_cache:
            try:
                with open(cache_file, "wb") as fh:
                    pickle.dump({"signature": signature, "headers": headers, "spans": spans},
                                fh, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as exc:  # Silently degrade for read-only directories, etc.
                log("  (Could not write index cache; ignoring: %s)" % exc)

    # -- Access -------------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.headers)

    def header(self, i: int) -> str:
        return self.headers[i]

    def sequence(self, i: int) -> str:
        override = self._overrides.get(i)
        if override is not None:
            return override
        cached = self._seq_cache.get(i)
        if cached is not None:
            self._seq_cache.move_to_end(i)
            return cached
        if self._fh is None:
            self._fh = open(self.path, "rb")
        start, end = self.spans[i]
        self._fh.seek(start)
        raw = self._fh.read(end - start)
        seq = raw.replace(b"\n", b"").replace(b"\r", b"").decode("ascii", "replace").upper()
        if len(seq) <= self._cache_limit:
            self._seq_cache[i] = seq
            self._cache_bytes += len(seq)
            while self._cache_bytes > self._cache_limit and self._seq_cache:
                _, evicted = self._seq_cache.popitem(last=False)
                self._cache_bytes -= len(evicted)
        return seq

    def set_overrides(self, overrides: Dict[int, str]) -> None:
        """Temporarily replace transcript sequences (MATLAB washed_tp().Sequence assignment)."""
        self._overrides = dict(overrides)

    def clear_overrides(self) -> None:
        self._overrides = {}

    def find_transcript(self, transcript_id: str) -> int:
        """Return the first record index whose Header contains the ID, or -1 (MATLAB contains loop)."""
        for i, header in enumerate(self.headers):
            if transcript_id in header:
                return i
        return -1

    def find_gene(self, gene_namefull: str) -> List[int]:
        """Return all record indices whose Header contains ``|GeneName|``."""
        return [i for i, header in enumerate(self.headers) if gene_namefull in header]

    def lookup_header_key(self, key: str) -> int:
        """Find a reference transcript by the first 44 Header characters, or return -1."""
        hit = self._key_map.get(key[:HEADER_KEY_LEN])
        if hit is not None:
            return hit
        # Fallback: for BLAST sseqid values shorter than 44 characters, use substring
        # matching (as MATLAB contains does).
        for i, header in enumerate(self.headers):
            if key in header[:HEADER_KEY_LEN]:
                return i
        return -1

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


# --------------------------------------------------------------------------- #
# 2. transcript_split.m — split transcripts into candidate probes.
# --------------------------------------------------------------------------- #
def reverse_complement(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


def iter_transcript_split(sequence: str, seq_id: str, binding_length: int,
                          fanxiang: int, mid_length: int
                          ) -> Iterator[Dict[str, object]]:
    """Yield candidate probes one at a time (the generator used by the main workflow).

    ``start`` retains MATLAB's one-based position, which is written to the output FASTA
    header and read back from BLAST results. The generator controls memory: the longest
    human reference transcript has 350,000 nt, and materializing it as a list costs ~255 MB.
    """
    half = binding_length // 2
    mid_half = mid_length // 2
    num_probe = len(sequence) - binding_length + 1
    rc = fanxiang == 1
    for i in range(num_probe):
        full = sequence[i:i + binding_length]
        left = sequence[i:i + half]
        right = sequence[i + half:i + binding_length]
        mid = sequence[i + half - mid_half:i + half + mid_half]
        if rc:
            full = reverse_complement(full)
            left = reverse_complement(left)
            right = reverse_complement(right)
            mid = reverse_complement(mid)
        yield {"full": full, "start": i + 1, "left": left,
               "right": right, "mid": mid, "ID": seq_id}


def transcript_split(sequence: str, seq_id: str, binding_length: int,
                     fanxiang: int, mid_length: int) -> List[Dict[str, object]]:
    """:func:`iter_transcript_split` as a list (for standalone calls and tests)."""
    return list(iter_transcript_split(sequence, seq_id, binding_length,
                                      fanxiang, mid_length))


def count_transcript_split(sequence_length: int, binding_length: int) -> int:
    """Return candidate count = transcript length - binding length + 1 (or 0 if insufficient)."""
    return max(0, sequence_length - binding_length + 1)


# --------------------------------------------------------------------------- #
# 3. probe_filter1.m — filter N / GC / Tm / homopolymers / hairpins / dimers.
# --------------------------------------------------------------------------- #
def probe_filter1(probes: Iterable[Dict[str, object]], min_tm: float, max_tm: float,
                  min_gc: float, max_gc: float, maxdm: int, maxstem: int,
                  poly_re: "re.Pattern") -> List[Dict[str, object]]:
    """Initial filtering: remove candidates containing N, then filter by GC / Tm,
    homopolymers, and secondary structure.

    ``probes`` may be a list or generator; it is traversed once without retaining an
    intermediate full list.

    Discard every candidate containing N (or any non-ACGT character): its GC/Tm can only
    be estimated as "half G/C, half A/T" and the synthesized sequence would be unusable.
    MATLAB's sequence_word2digital raises on N, so the default candidate set contains none.
    """
    # MATLAB uses two passes (GC/Tm, then homopolymers + secondary structure); each pass
    # evaluates records independently. Combining them preserves results and order while
    # avoiding a second full intermediate list.
    n_dropped = 0
    kept: List[Dict[str, object]] = []
    for probe in probes:
        full = probe["full"]                                # type: ignore[assignment]
        if not _ACGT_ONLY_RE.match(full):                   # type: ignore[arg-type]
            n_dropped += 1
            continue
        gc, tm = gc_and_mean_tm(probe["left"])              # type: ignore[arg-type]
        if not (min_gc <= gc <= max_gc and min_tm <= tm <= max_tm):
            continue
        gc, tm = gc_and_mean_tm(probe["right"])             # type: ignore[arg-type]
        if not (min_gc <= gc <= max_gc and min_tm <= tm <= max_tm):
            continue
        if poly_re.search(full):                            # type: ignore[arg-type]
            continue
        has_hairpin, has_dimer = check_hairpin_dimer(full, maxstem, maxdm)  # type: ignore[arg-type]
        if has_hairpin or has_dimer:
            continue
        kept.append(probe)
    if n_dropped:
        log("  Dropped %d candidates containing N/non-ACGT bases" % n_dropped)
    return kept


# --------------------------------------------------------------------------- #
# 4. probe_2_txt_targer_region.m / daochu_targer_region.m — export FASTA.
# --------------------------------------------------------------------------- #
def write_target_region(path: str, probes: Sequence[Dict[str, object]],
                        append: bool = False) -> None:
    """Write ``>|<start>|<header>`` plus sequence, byte-for-byte matching MATLAB."""
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8", newline="\n") as fh:
        for probe in probes:
            fh.write(">|%d|%s\n%s\n" % (probe["start"], probe["ID"], probe["full"]))


# --------------------------------------------------------------------------- #
# 5. BLAST
# --------------------------------------------------------------------------- #
def find_blast_executable(name: str, blast_bin: Optional[str]) -> str:
    exe = name + (".exe" if os.name == "nt" else "")
    if blast_bin:
        candidate = os.path.join(blast_bin, exe)
        if os.path.isfile(candidate):
            return candidate
        raise FileNotFoundError("Could not find %s in the directory specified by --blast-bin: %s"
                                % (exe, blast_bin))
    found = shutil.which(name)
    if found:
        return found
    guesses = ["/usr/local/ncbi/blast/bin", "/usr/bin", "/usr/local/bin",
               "/opt/homebrew/bin", r"C:\Program Files\NCBI\blast\bin"]
    guesses += sorted(glob.glob(r"C:\Program Files\NCBI\blast-*\bin"), reverse=True)
    for directory in guesses:
        candidate = os.path.join(directory, exe)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        "Could not find %s. Add the NCBI BLAST+ bin directory to PATH or use --blast-bin."
        % name)


def resolve_blast_db(db: str, data_dir: str) -> str:
    """Resolve a database name to a path prefix that blastn can use directly."""
    if os.path.dirname(db):
        return os.path.abspath(db)
    candidate = os.path.join(data_dir, db)
    for suffix in (".nin", ".nal"):
        if os.path.isfile(candidate + suffix):
            return os.path.abspath(candidate)
    return db  # Let the BLASTDB environment variable resolve it.


def check_blast_db(db_prefix: str) -> bool:
    return any(os.path.isfile(db_prefix + s) for s in (".nin", ".nal"))


def make_blast_db(makeblastdb: str, fasta: str, db_prefix: str) -> None:
    log("  BLAST database not found; building it with %s (one-time step, may take minutes)..."
        % os.path.basename(fasta))
    cmd = [makeblastdb, "-in", fasta, "-dbtype", "nucl", "-out", db_prefix]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError("makeblastdb failed:\n%s" % proc.stdout.decode("utf-8", "replace"))
    log("  Database build complete: %s" % db_prefix)


def run_blastn(blastn: str, query: str, db_prefix: str, evalue: float,
               threads: int, task: str, out_path: Optional[str] = None,
               min_free_bytes: int = 0,
               stats: Optional[Dict[str, int]] = None) -> Iterator[str]:
    """Run blastn and **yield outfmt 6 results line by line**.

    When ``out_path`` is None (the default), results only pass through the pipe and no
    bytes are written to disk;
    this keeps temporary storage bounded for long transcripts.
    when ``out_path`` is provided, also write a copy while reading (for
    ``--keep-blast-out``), checking free disk space every 64 MB and aborting below
    ``min_free_bytes`` to avoid filling the disk.
    """
    cmd = [blastn, "-query", query, "-db", db_prefix, "-task", task,
           "-evalue", "%g" % evalue, "-outfmt", "6", "-num_threads", str(threads)]
    err_file = tempfile.TemporaryFile()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err_file,
                            bufsize=1 << 20)
    sink = open(out_path, "w", encoding="utf-8", newline="\n") if out_path else None
    written = 0
    next_check = 1 << 26     # 64 MB
    try:
        stream = io.TextIOWrapper(proc.stdout, encoding="utf-8", errors="replace")
        for line in stream:
            if sink is not None:
                sink.write(line)
                written += len(line)
                if written >= next_check:
                    next_check = written + (1 << 26)
                    if free_disk_bytes(out_path) < min_free_bytes:
                        raise RuntimeError(
                            "Insufficient free disk space (%.1f GB of raw BLAST output retained). "
                            "Remove --keep-blast-out to avoid writing to disk."
                            % (written / 1073741824.0))
            yield line
        proc.wait()
        if proc.returncode != 0:
            err_file.seek(0)
            raise RuntimeError("blastn failed (returncode=%d):\n%s"
                               % (proc.returncode,
                                  err_file.read().decode("utf-8", "replace")[-4000:]))
        if stats is not None:
            stats["bytes"] = written
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        try:
            proc.stdout.close()
        except (OSError, AttributeError):
            pass
        err_file.close()
        if sink is not None:
            sink.close()


# --------------------------------------------------------------------------- #
# 6. read_and_clear_probe_txt.m — read and clean BLAST results.
# --------------------------------------------------------------------------- #
_PROBE_START_RE = re.compile(r"\d+")


def _parse_probe_start(qseqid: str) -> Optional[int]:
    """Extract probe start 125 from a query ID such as ``|125|ENSMUST...``."""
    parts = [p for p in qseqid.split("|") if p]
    if parts and parts[0].isdigit():
        return int(parts[0])
    match = _PROBE_START_RE.search(qseqid)
    return int(match.group()) if match else None


def iter_blast_hits(lines: Iterable[str], index: FastaIndex, gene_namefull: str,
                    singlehomo: int, binding_length: int,
                    stats: Optional[Dict[str, int]] = None
                    ) -> Iterator[Tuple[int, int, int, int]]:
    """Clean BLAST results line by line and yield
    ``(probe start, reference transcript index, match start, match end)``.

    Apply the MATLAB filters: remove hits to the target gene itself, query alignments
    shorter than singlehomo, and reverse (complementary) subject alignments. Compute
    endpoints by aligning probe position 1 to the reference sequence.

    This is a generator rather than a list: one Ccr5 gene can have 320,000 hits and a
    very long transcript can have tens of millions, which would require gigabytes as a
    list. ``stats`` is populated with ``kept`` / ``unresolved`` counts.
    """
    kept = 0
    unresolved = 0
    for line in lines:
        line = line.rstrip("\r\n")
        if not line:
            continue
        fields = line.split("\t")
        if len(fields) < 12:
            continue
        sseqid = fields[1]
        if gene_namefull in sseqid:          # Match to the target RNA (including isoforms).
            continue
        try:
            qstart, qend = int(fields[6]), int(fields[7])
            sstart, send = int(fields[8]), int(fields[9])
        except ValueError:
            continue
        if (qend - qstart + 1) < singlehomo:  # Homologous segment too short.
            continue
        if send < sstart:                     # Complementary-strand alignment.
            continue
        probe_start = _parse_probe_start(fields[0])
        if probe_start is None:
            continue
        ref_index = index.lookup_header_key(sseqid)
        if ref_index < 0:
            unresolved += 1
            continue
        start = sstart - qstart + 1 if sstart >= qstart else 1
        kept += 1
        yield (probe_start, ref_index, start, start + binding_length - 1)
    if stats is not None:
        stats["kept"] = kept
        stats["unresolved"] = unresolved


def read_and_clear_blast(index: FastaIndex, blast_out: str, gene_namefull: str,
                         singlehomo: int, binding_length: int
                         ) -> Tuple[List[Tuple[int, int, int, int]], int]:
    """:func:`iter_blast_hits` as a list (reads an existing BLAST file for standalone use and tests)."""
    if not os.path.isfile(blast_out):
        return [], 0
    stats: Dict[str, int] = {}
    with open(blast_out, "r", encoding="utf-8", errors="replace") as fh:
        rows = list(iter_blast_hits(fh, index, gene_namefull, singlehomo,
                                    binding_length, stats))
    return rows, stats.get("unresolved", 0)


# --------------------------------------------------------------------------- #
# 7. probe_nonspecific_binding_check_hamming.m — final nonspecific-binding filter.
# --------------------------------------------------------------------------- #
def nonspecific_binding_check_hamming(probes: Sequence[Dict[str, object]],
                                      get_sequence: Callable[[int], str],
                                      blast_rows: Iterable[Tuple[int, int, int, int]],
                                      binding_length: int, singlehomo: int,
                                      mid_length: int, bothhomo: int
                                      ) -> List[Dict[str, object]]:
    """Compare positions using Hamming distance and remove probes with excessive
    cross-transcript homology.

    ``blast_rows`` may be a list or generator (the main workflow consumes the blastn
    pipe directly). Each hit only marks a probe unusable, so order does not matter and
    streaming is equivalent.

    Two exclusion rules matching MATLAB:
      1) total matches >= bothhomo, both sides >= singlehomo, and matches in the
         central (mid_length+2) nt window > 2;
      2) total matches >= bothhomo, either side >= singlehomo, at least one match in
         the central 2 nt, and each adjacent 6 nt window has > 4 matches.
    If the reference sequence is out of bounds or contains N, MATLAB retains the
    preallocated all-one row (treating it as fully homologous and excluding strictly);
    this implementation does the same.
    """
    half = binding_length // 2
    mid_half = mid_length // 2
    lo_mid, hi_mid = half - mid_half - 1, half + mid_half + 1

    start_to_index: Dict[int, int] = {}
    for i, probe in enumerate(probes):
        start_to_index.setdefault(probe["start"], i)   # type: ignore[arg-type]

    all_match = [1] * binding_length
    keep = [True] * len(probes)

    for probe_start, ref_index, seg_start, seg_end in blast_rows:
        probe_index = start_to_index.get(probe_start)
        if probe_index is None:        # MATLAB blast_probe_xuliehao == 0: skip directly.
            continue
        if not keep[probe_index]:
            continue

        ref_seq = get_sequence(ref_index)
        if len(ref_seq) < seg_end:
            chabie = all_match
        else:
            segment = ref_seq[seg_start - 1:seg_end]
            if "N" in segment:
                chabie = all_match
            else:
                probe_seq = probes[probe_index]["full"]     # type: ignore[assignment]
                chabie = [1 if segment[k] == probe_seq[k] else 0
                          for k in range(binding_length)]

        if sum(chabie) < bothhomo:
            continue
        left_match = sum(chabie[:half])
        right_match = sum(chabie[half:binding_length])
        if (left_match >= singlehomo and right_match >= singlehomo
                and sum(chabie[lo_mid:hi_mid]) > 2):
            keep[probe_index] = False
            continue
        if (left_match >= singlehomo or right_match >= singlehomo):
            if (sum(chabie[half - 1:half + 1]) > 0
                    and sum(chabie[half - 6:half]) > 4
                    and sum(chabie[half:half + 6]) > 4):
                keep[probe_index] = False

    return [probe for i, probe in enumerate(probes) if keep[i]]


def apply_gaplength(probes: Sequence[Dict[str, object]], gaplength: int
                    ) -> List[Dict[str, object]]:
    """Optional: ensure adjacent probes for one gene are at least gaplength nt apart
    (disabled in MATLAB)."""
    kept: List[Dict[str, object]] = []
    last_end = None
    for probe in sorted(probes, key=lambda p: p["start"]):     # type: ignore[arg-type]
        start = probe["start"]                                  # type: ignore[assignment]
        if last_end is not None and start - last_end - 1 < gaplength:
            continue
        kept.append(probe)
        last_end = start + len(probe["full"]) - 1               # type: ignore[arg-type]
    return kept


# --------------------------------------------------------------------------- #
# 8. Complete workflow for one transcript.
# --------------------------------------------------------------------------- #
class DesignResult(object):
    def __init__(self, gene_name: str, transcript_id: str):
        self.gene_name = gene_name
        self.transcript_id = transcript_id
        self.status = "pending"
        self.n_first = 0
        self.n_second = 0
        self.n_final = 0
        self.result_file = ""
        self.seconds = 0.0


def _safe_name(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", str(name).strip())


def design_one(index: FastaIndex, gene_name: str, transcript_id: str,
               params: Dict[str, object], paths: Dict[str, str],
               blastn: str, threads: int, append: bool,
               poly_re: "re.Pattern", blast_task: str = "blastn-short",
               keep_blast_out: bool = False) -> DesignResult:
    result = DesignResult(gene_name, transcript_id)

    binding_length = int(params["binding_length"])
    mid_length = int(params["mid_length"])
    evalue = float(params["evalue"])
    gene_namefull = "|%s|" % gene_name

    safe_gene = _safe_name(gene_name)
    safe_id = _safe_name(transcript_id)
    query_file = os.path.join(paths["tmp"], "%s_%s_qianzai_target_region_seq.txt"
                              % (safe_gene, safe_id))
    blast_file = os.path.join(paths["tmp"], "%s_%s_e%s.txt"
                              % (safe_gene, safe_id, "%g" % evalue))
    result_file = os.path.join(paths["result"], "%s_%s_target_region_seq.txt"
                               % (safe_gene, safe_id))

    # ---- Locate target transcript ----------------------------------------- #
    target_index = index.find_transcript(transcript_id)
    if target_index < 0:
        log("  ID not found; skipping")
        result.status = "id_not_found"
        return result
    log("  ID found: %s" % index.header(target_index))

    # ---- Isoform handling: as in MATLAB, replace other isoforms of the same gene with
    #      the first sequence in the reference so they are no longer "homologous" in
    #      subsequent Hamming comparisons. --------------------------------- #
    overrides = {target_index: TARGET_MASK_SEQ}   # Mask the target RNA itself.
    if int(params["isoform_spec"]) == 0:
        filler = index.sequence(0)
        for i in index.find_gene(gene_namefull):
            if i != target_index:
                overrides[i] = filler

    target_sequence = index.sequence(target_index)
    target_header = index.header(target_index)

    # ---- (1/7) Split candidate probes (generator; do not materialize all) --- #
    result.n_first = count_transcript_split(len(target_sequence), binding_length)
    if result.n_first == 0:
        log("  Step 1 failed: cannot design (transcript length %d < binding region %d)"
            % (len(target_sequence), binding_length))
        result.status = "split_failed"
        return result
    log("  Initial split complete: %d candidates" % result.n_first)

    # ---- (2/7) GC / Tm / homopolymers / secondary structure ---------------- #
    probe_second_list = probe_filter1(
        iter_transcript_split(target_sequence, target_header, binding_length,
                              int(params["fanxiang"]), mid_length),
        float(params["min_tm"]), float(params["max_tm"]), float(params["min_gc"]),
        float(params["max_gc"]), int(params["maxdm"]), int(params["maxstem"]), poly_re)
    result.n_second = len(probe_second_list)
    if not probe_second_list:
        log("  Step 2 failed: no design")
        result.status = "filter_failed"
        return result
    log("  Secondary filtering complete: %d candidates" % result.n_second)

    max_candidates = int(params.get("max_candidates") or 0)
    if max_candidates and result.n_second > max_candidates:
        log("  Candidate count %d exceeds --max-candidates %d; skipping gene"
            " (estimated BLAST output %.1f GB, runtime %.0f minutes)"
            % (result.n_second, max_candidates,
               result.n_second * 0.216 / 1024.0, result.n_second * 0.08 / 60.0))
        result.status = "too_many_candidates"
        return result

    # ---- (3/7) Export potential probe sequences --------------------------- #
    write_target_region(query_file, probe_second_list, append=append)
    log("  Potential probe sequences exported: %s" % query_file)

    # ---- (4-6/7) BLAST -> clean -> final nonspecific filter, one pipeline -- #
    #      BLAST results stream into memory line by line: by default no bytes are
    #      written to disk and hit rows are not accumulated, so disk and memory use
    #      stay constant regardless of transcript length.
    min_free_bytes = int(params.get("min_free_bytes") or 0)
    if free_disk_bytes(paths["tmp"]) < min_free_bytes:
        raise RuntimeError("Only %.1f GB remains on the output disk; refusing to start"
                           % (min_free_bytes / 1073741824.0))

    t_blast = time.time()
    blast_stats: Dict[str, int] = {}
    hit_stats: Dict[str, int] = {}
    lines = run_blastn(blastn, query_file, paths["db"], evalue, threads, blast_task,
                       out_path=blast_file if keep_blast_out else None,
                       min_free_bytes=min_free_bytes, stats=blast_stats)
    blast_rows = iter_blast_hits(lines, index, gene_namefull,
                                 int(params["singlehomo"]), binding_length, hit_stats)

    index.set_overrides(overrides)
    try:
        probe_third_list = nonspecific_binding_check_hamming(
            probe_second_list, index.sequence, blast_rows, binding_length,
            int(params["singlehomo"]), mid_length, int(params["bothhomo"]))
    finally:
        index.clear_overrides()

    log("  BLAST + cleaning + final filtering complete in %.1f s (%d sites checked%s)"
        % (time.time() - t_blast, hit_stats.get("kept", 0),
           ", %.0f MB of raw BLAST output retained" % (blast_stats.get("bytes", 0) / 1048576.0)
           if keep_blast_out else ", raw BLAST output not written to disk"))
    if hit_stats.get("unresolved"):
        log("  Note: %d BLAST results had no matching reference transcript (ignored)"
            % hit_stats["unresolved"])

    if params.get("apply_gaplength"):
        probe_third_list = apply_gaplength(probe_third_list, int(params["gaplength"]))

    result.n_final = len(probe_third_list)
    if not probe_third_list:
        log("  Step 3 failed: no design; adjust parameters")
        result.status = "nonspecific_failed"
        return result
    log("  Final filtering complete: %d target regions" % result.n_final)

    # ---- (7/7) Export target regions -------------------------------------- #
    write_target_region(result_file, probe_third_list, append=append)
    result.result_file = result_file
    result.status = "ok"
    log("  Target region sequences exported: %s" % result_file)
    return result


# --------------------------------------------------------------------------- #
# 9. Read gene lists (txt / tsv / csv / xlsx).
# --------------------------------------------------------------------------- #
_ENS_RE = re.compile(r"^ENS[A-Z]*[TG]?\d{6,}", re.IGNORECASE)

_ID_ALIASES = ("transcript_id", "transcriptid", "transcript", "ensembl_id",
               "ensembl", "id", "gene_id", "geneid", "\u8f6c\u5f55\u672c", "\u8f6c\u5f55\u672cid")
_NAME_ALIASES = ("gene_name", "genename", "gene", "symbol", "name", "\u57fa\u56e0", "\u57fa\u56e0\u540d")


def _read_xlsx_rows(path: str, sheet: object = 0) -> List[List[str]]:
    """Minimal xlsx reader using only standard-library zipfile + XML; enough for two ID/name columns."""
    import xml.etree.ElementTree as ET

    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    rel_ns = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
    with zipfile.ZipFile(path) as zf:
        shared: List[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall(ns + "si"):
                shared.append("".join(t.text or "" for t in si.iter(ns + "t")))

        wb = ET.fromstring(zf.read("xl/workbook.xml"))
        sheets = wb.find(ns + "sheets").findall(ns + "sheet")
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {r.get("Id"): r.get("Target") for r in rels}

        chosen = None
        for i, node in enumerate(sheets):
            if (isinstance(sheet, int) and i == sheet) or \
               (isinstance(sheet, str) and node.get("name") == sheet):
                chosen = node
                break
        if chosen is None:
            raise ValueError("Sheet not found in xlsx: %r" % (sheet,))
        target = rel_map[chosen.get(rel_ns + "id")].lstrip("/")
        if not target.startswith("xl/"):
            target = "xl/" + target

        rows: List[List[str]] = []
        sheet_root = ET.fromstring(zf.read(target))
        for row in sheet_root.iter(ns + "row"):
            values: List[str] = []
            for cell in row.findall(ns + "c"):
                ctype = cell.get("t")
                if ctype == "inlineStr":
                    node = cell.find(ns + "is")
                    text = "".join(t.text or "" for t in node.iter(ns + "t")) if node is not None else ""
                else:
                    node = cell.find(ns + "v")
                    text = node.text or "" if node is not None else ""
                    if ctype == "s" and text.isdigit():
                        text = shared[int(text)]
                values.append(text)
            rows.append(values)
        return rows


def _looks_like_id(value: str) -> bool:
    return bool(_ENS_RE.match(value.strip()))


def read_gene_list(path: str, id_col: Optional[str], name_col: Optional[str],
                   sheet: object) -> List[Tuple[str, str]]:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xlsm"):
        rows = _read_xlsx_rows(path, sheet)
    else:
        with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as fh:
            text = fh.read()
        delimiter = "," if ext == ".csv" else ("\t" if "\t" in text else
                                               ("," if "," in text else None))
        if delimiter is None:
            rows = [line.split() for line in text.splitlines()]
        else:
            rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))

    rows = [[str(c).strip() for c in row] for row in rows if any(str(c).strip() for c in row)]
    if not rows:
        raise ValueError("Gene list is empty: %s" % path)

    header_row: Optional[List[str]] = None
    if not _looks_like_id(rows[0][0] if rows[0] else "") and len(rows) > 1:
        header_row = [c.lower() for c in rows[0]]
        rows = rows[1:]

    def resolve(spec: Optional[str], aliases: Sequence[str], fallback: int) -> int:
        if spec is not None:
            if spec.isdigit():
                return int(spec) - 1          # The user supplied a one-based column number.
            if header_row and spec.lower() in header_row:
                return header_row.index(spec.lower())
            raise ValueError("Column %r not found in gene list" % spec)
        if header_row:
            for alias in aliases:
                if alias in header_row:
                    return header_row.index(alias)
        return fallback

    id_index = resolve(id_col, _ID_ALIASES, 0)
    name_index = resolve(name_col, _NAME_ALIASES, 1)

    pairs: List[Tuple[str, str]] = []
    for row in rows:
        if len(row) <= max(id_index, name_index):
            continue
        transcript_id, gene_name = row[id_index], row[name_index]
        if not transcript_id or not gene_name:
            continue
        # Correct reversed columns automatically (MATLAB fixes column 1 = ID, column 2 = name).
        if not _looks_like_id(transcript_id) and _looks_like_id(gene_name):
            transcript_id, gene_name = gene_name, transcript_id
        pairs.append((transcript_id, gene_name))
    if not pairs:
        raise ValueError("Could not parse any (transcript_id, gene_name) pairs from %s" % path)
    return pairs


# --------------------------------------------------------------------------- #
# 10. CLI.
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="probe_design.py",
        description="BASSFISH probe target region design (Python version of target_region_matlab)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    group = parser.add_argument_group("Input")
    group.add_argument("--species", choices=sorted(SPECIES_PRESET),
                       help="Species preset: human -> hdb + gencode_v42_transcripts.fasta; "
                            "mouse -> mdb + gencode.vM26.transcripts.fasta")
    group.add_argument("--data-dir", default=None,
                       help="Directory containing transcript FASTA and BLAST database "
                            "(auto-detect target_region_matlab by default)")
    group.add_argument("--ref", default=None, help="Transcript FASTA path (overrides the --species default)")
    group.add_argument("--db", default=None, help="BLAST database name or path prefix (overrides --species)")
    group.add_argument("--genes", default=None,
                       help="Gene-list file: txt/tsv/csv/xlsx; column 1 transcript_id, column 2 gene_name")
    group.add_argument("--gene", action="append", default=[], metavar="ID,NAME",
                        help="Specify one gene directly, e.g. --gene ENSMUST00000111442,Ccr5 (repeatable)")
    group.add_argument("--id-col", default=None, help="Transcript ID column name or number (one-based)")
    group.add_argument("--name-col", default=None, help="Gene name column name or number (one-based)")
    group.add_argument("--sheet", default="0", help="xlsx sheet name or index")

    group = parser.add_argument_group("Output")
    group.add_argument("-o", "--output", default="./out",
                       help="Output directory; tmp/ and result/ are created inside it")
    group.add_argument("--append", action="store_true",
                       help="Append to existing output files (MATLAB uses a+); overwrite by default")
    group.add_argument("--summary", default="summary.csv",
                       help="CSV filename for per-gene statistics (empty disables writing)")
    group.add_argument("--keep-blast-out", action="store_true",
                       help="Write raw BLAST results to tmp/; by default stream through a pipe "
                            "without writing (hundreds of MB per gene, tens of GB for long transcripts)")

    group = parser.add_argument_group("Design parameters (same names as MATLAB variables)")
    group.add_argument("--fanxiang", type=int, choices=(0, 1), default=DEFAULTS["fanxiang"],
                       help="1 = reverse-complement probes, 0 = transcript orientation")
    group.add_argument("--isoform-spec", type=int, choices=(0, 1),
                        default=DEFAULTS["isoform_spec"], help="Whether to account for isoform specificity")
    group.add_argument("--binding-length", type=int, default=DEFAULTS["binding_length"],
                       help="Binding-region length (must be even)")
    group.add_argument("--looooong", type=int, default=DEFAULTS["looooong"],
                       help="Exclude homopolymers; 5 excludes GGGGG, for example")
    group.add_argument("--min-tm", type=float, default=DEFAULTS["min_tm"])
    group.add_argument("--max-tm", type=float, default=DEFAULTS["max_tm"])
    group.add_argument("--min-gc", type=float, default=DEFAULTS["min_gc"])
    group.add_argument("--max-gc", type=float, default=DEFAULTS["max_gc"])
    group.add_argument("--maxdm", type=int, default=DEFAULTS["maxdm"], help="Maximum dimer length")
    group.add_argument("--maxstem", type=int, default=DEFAULTS["maxstem"], help="Maximum hairpin stem length")
    group.add_argument("--mid-length", type=int, default=DEFAULTS["mid_length"],
                       help="Central mismatch-window length (must be even)")
    group.add_argument("--singlehomo", type=int, default=DEFAULTS["singlehomo"],
                       help="Maximum allowed one-sided homology")
    group.add_argument("--bothhomo", type=int, default=DEFAULTS["bothhomo"],
                       help="Maximum allowed homology for the full probe")
    group.add_argument("--gaplength", type=int, default=DEFAULTS["gaplength"],
                       help="Minimum gap between adjacent probes for one gene (use with --apply-gaplength)")
    group.add_argument("--apply-gaplength", action="store_true",
                       help="Enable gaplength filtering (declared but unused by MATLAB)")
    group.add_argument("--evalue", type=float, default=DEFAULTS["evalue"], help="blastn e-value")

    group = parser.add_argument_group("Runtime / resource limits")
    group.add_argument("--blast-bin", default=None,
                       help="NCBI BLAST+ bin directory (search PATH and common locations by default)")
    group.add_argument("--threads", type=int, default=0,
                       help="blastn -num_threads; 0 = min(%d, detected available CPUs), "
                            "never exceeding the available CPU count" % DEFAULT_MAX_THREADS)
    group.add_argument("--max-threads", type=int, default=DEFAULT_MAX_THREADS,
                       help="Hard thread limit")
    group.add_argument("--seq-cache-mb", type=int, default=DEFAULT_SEQ_CACHE_MB,
                       help="Reference-sequence LRU cache memory limit (MB)")
    group.add_argument("--min-free-gb", type=float, default=DEFAULT_MIN_FREE_GB,
                       help="Refuse to start or abort if free disk space falls below this value")
    group.add_argument("--max-candidates", type=int, default=0,
                       help="Skip a gene when post-filter candidates exceed this value (0 = unlimited); "
                            "a safeguard against the large BLAST load from long transcripts")
    group.add_argument("--lock-file", default=None,
                       help="use this file as a process lock")
    group.add_argument("--lock-timeout", type=float, default=1800.0,
                       help="Maximum seconds to wait for the run lock; error on timeout")
    group.add_argument("--blast-task", default="blastn-short",
                       help="blastn -task (fixed to blastn-short by MATLAB)")
    group.add_argument("--makeblastdb", action="store_true",
                       help="Build the database automatically from --ref if it does not exist")
    group.add_argument("--no-index-cache", action="store_true",
                       help="Do not read or write the FASTA index cache (<ref>.pdidx)")
    group.add_argument("--stop-on-error", action="store_true",
                       help="Stop immediately on any gene error (skip that gene and continue by default)")
    group.add_argument("--version", action="version", version="%(prog)s " + __version__)
    return parser


def _dir_size_mb(path: str) -> float:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total / 1048576.0


def guess_data_dir() -> str:
    for candidate in (os.path.join(_HERE, "..", "target_region_matlab"),
                      os.path.join(os.getcwd(), "target_region_matlab"),
                      os.getcwd()):
        candidate = os.path.abspath(candidate)
        if os.path.isdir(candidate):
            return candidate
    return os.getcwd()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # ---- Validate parameters ---------------------------------------------- #
    if args.binding_length % 2 != 0:
        parser.error("--binding-length must be even")
    if args.mid_length % 2 != 0:
        parser.error("--mid-length must be even")
    half = args.binding_length // 2
    if half < 6:
        parser.error("--binding-length must be at least 12 (final filtering uses a central +/-6 nt window)")
    if half - args.mid_length // 2 - 1 < 0:
        parser.error("--mid-length is too large relative to --binding-length")
    if args.maxstem <= 1 or args.maxstem >= args.binding_length:
        parser.error("--maxstem must be between 1 and binding_length")
    if args.maxdm <= 1 or args.maxdm >= args.binding_length:
        parser.error("--maxdm must be between 1 and binding_length")

    # ---- Resource limits: converge on cgroup limits because containers report host CPUs. ---- #
    cpu_limit = detect_cpu_limit()
    ceiling = max(1, min(args.max_threads, cpu_limit))
    if args.threads <= 0:
        threads = ceiling
    elif args.threads > ceiling:
        log("Notice: --threads %d exceeds the limit (available CPUs %d, --max-threads %d); using %d"
            % (args.threads, cpu_limit, args.max_threads, ceiling))
        threads = ceiling
    else:
        threads = args.threads
    min_free_bytes = int(max(0.0, args.min_free_gb) * 1073741824)

    data_dir = os.path.abspath(args.data_dir) if args.data_dir else guess_data_dir()

    ref = args.ref
    db = args.db
    if args.species:
        preset = SPECIES_PRESET[args.species]
        ref = ref or os.path.join(data_dir, preset["fasta"])
        db = db or preset["db"]
    if not ref or not db:
        parser.error("Choose human/mouse with --species, or specify both --ref and --db manually")
    ref = os.path.abspath(ref)

    # ---- Gene list --------------------------------------------------------- #
    gene_pairs: List[Tuple[str, str]] = []
    for item in args.gene:
        parts = [p.strip() for p in re.split(r"[,\t;]", item) if p.strip()]
        if len(parts) != 2:
            parser.error("--gene requires two fields, 'transcript_id,gene_name'; received: %r" % item)
        transcript_id, gene_name = parts
        if not _looks_like_id(transcript_id) and _looks_like_id(gene_name):
            transcript_id, gene_name = gene_name, transcript_id
        gene_pairs.append((transcript_id, gene_name))
    if args.genes:
        sheet: object = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
        gene_pairs.extend(read_gene_list(args.genes, args.id_col, args.name_col, sheet))
    if not gene_pairs:
        parser.error("Provide genes with --genes or --gene")

    # ---- Output directories ------------------------------------------------ #
    output = os.path.abspath(args.output)
    paths = {"out": output,
             "tmp": os.path.join(output, "tmp"),
             "result": os.path.join(output, "result")}
    for directory in paths.values():
        os.makedirs(directory, exist_ok=True)

    # ---- Disk preflight --------------------------------------------------- #
    free_gb = free_disk_bytes(output) / 1073741824.0
    if free_gb < args.min_free_gb:
        raise RuntimeError("Only %.1f GB is available on the output disk, below --min-free-gb %.1f; refusing to start"
                           % (free_gb, args.min_free_gb))

    # ---- BLAST environment ------------------------------------------------ #
    blastn = find_blast_executable("blastn", args.blast_bin)
    db_prefix = resolve_blast_db(db, data_dir)
    if not check_blast_db(db_prefix):
        if args.makeblastdb:
            makeblastdb = find_blast_executable("makeblastdb", args.blast_bin)
            if os.path.dirname(db_prefix) == "":
                db_prefix = os.path.join(data_dir, db_prefix)
            make_blast_db(makeblastdb, ref, db_prefix)
        else:
            log("Warning: no BLAST database file found in %s (%s.nin); "
                "passing the name directly to blastn (use --makeblastdb to build it automatically)"
                % (data_dir, db_prefix))
    paths["db"] = db_prefix

    log("=" * 72)
    log("BASSFISH probe target region design (Python v%s)" % __version__)
    log("  Species/database : %s / %s" % (args.species or "-", db_prefix))
    log("  Transcript reference : %s" % ref)
    log("  Gene count         : %d" % len(gene_pairs))
    log("  Output directory   : %s" % output)
    log("  blastn      : %s (threads=%d, evalue=%g)" % (blastn, threads, args.evalue))
    log("  Parameters         : binding_length=%d fanxiang=%d isoform_spec=%d looooong=%d "
        "Tm=[%g,%g] GC=[%g,%g] maxdm=%d maxstem=%d mid_length=%d "
        "singlehomo=%d bothhomo=%d"
        % (args.binding_length, args.fanxiang, args.isoform_spec, args.looooong,
           args.min_tm, args.max_tm, args.min_gc, args.max_gc, args.maxdm,
           args.maxstem, args.mid_length, args.singlehomo, args.bothhomo))
    mem_limit = detect_memory_limit_mb()
    log("  Resource limits    : %d CPUs available (%d used) | memory%s | %.1f GB disk free (minimum %.1f) "
        "| sequence cache %d MB"
        % (cpu_limit, threads,
           (" limit %d MB" % mem_limit) if mem_limit else " limit undetected",
           free_gb, args.min_free_gb, args.seq_cache_mb))
    log("  BLAST output      : %s"
        % ("retained in tmp/ (watch disk usage)" if args.keep_blast_out
           else "streamed through a pipe; not written to disk"))
    log("=" * 72)

    params = {
        "fanxiang": args.fanxiang, "isoform_spec": args.isoform_spec,
        "binding_length": args.binding_length, "min_tm": args.min_tm,
        "max_tm": args.max_tm, "min_gc": args.min_gc, "max_gc": args.max_gc,
        "maxdm": args.maxdm, "maxstem": args.maxstem, "mid_length": args.mid_length,
        "singlehomo": args.singlehomo, "bothhomo": args.bothhomo,
        "gaplength": args.gaplength, "apply_gaplength": args.apply_gaplength,
        "evalue": args.evalue, "max_candidates": args.max_candidates,
        "min_free_bytes": min_free_bytes,
    }
    poly = "A" * args.looooong + "|" + "T" * args.looooong + "|" + \
           "C" * args.looooong + "|" + "G" * args.looooong
    poly_re = re.compile(poly, re.IGNORECASE)

    log("Reading transcript data ...")
    t0 = time.time()
    index = FastaIndex(ref, use_cache=not args.no_index_cache,
                       cache_mb=args.seq_cache_mb)
    log("Read complete in %.1f s" % (time.time() - t0))

    results: List[DesignResult] = []
    run_lock = RunLock(args.lock_file, args.lock_timeout)
    try:
        run_lock.__enter__()
        for n, (transcript_id, gene_name) in enumerate(gene_pairs, 1):
            log("")
            log("[%d/%d] %s  %s" % (n, len(gene_pairs), gene_name, transcript_id))
            t_gene = time.time()
            try:
                result = design_one(index, gene_name, transcript_id, params, paths,
                                    blastn, threads, args.append, poly_re,
                                    args.blast_task, args.keep_blast_out)
            except Exception as exc:                      # noqa: BLE001
                log("  Error: %s" % exc)
                if args.stop_on_error:
                    raise
                result = DesignResult(gene_name, transcript_id)
                result.status = "error: %s" % exc
            result.seconds = time.time() - t_gene
            log("  Gene completed in %.1f s" % result.seconds)
            results.append(result)
    finally:
        run_lock.__exit__()
        index.close()
        # Fallback: an interrupted/error run may leave raw BLAST output; scan again after all genes.
        if not args.keep_blast_out:
            freed = 0
            for leftover in glob.glob(os.path.join(paths["tmp"], "*_e%s.txt" % ("%g" % args.evalue))):
                try:
                    freed += os.path.getsize(leftover)
                    os.remove(leftover)
                except OSError:
                    pass
            if freed:
                log("Removed leftover raw BLAST output, freeing %.0f MB" % (freed / 1048576.0))

    # ---- Summary ----------------------------------------------------------- #
    ok = [r for r in results if r.status == "ok"]
    log("")
    log("=" * 72)
    log("Final filtering complete for all genes: %d / %d succeeded" % (len(ok), len(results)))
    peak = peak_rss_mb()
    log("Resource usage: peak memory %s | output directory %.0f MB | disk still free %.1f GB"
        % ("%.0f MB" % peak if peak is not None else "unknown",
           _dir_size_mb(output), free_disk_bytes(output) / 1073741824.0))
    for result in results:
        if result.status != "ok":
            log("  No output: %-16s %-24s (%s)"
                % (result.gene_name, result.transcript_id, result.status))

    if args.summary:
        summary_path = os.path.join(output, args.summary)
        with open(summary_path, "w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["gene_name", "transcript_id", "status",
                             "n_candidate", "n_after_filter", "n_target_region",
                             "seconds", "result_file"])
            for result in results:
                writer.writerow([result.gene_name, result.transcript_id, result.status,
                                 result.n_first, result.n_second, result.n_final,
                                 "%.1f" % result.seconds, result.result_file])
        log("Summary written: %s" % summary_path)
    log("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("\nInterrupted")
        sys.exit(130)
    except (FileNotFoundError, ValueError, RuntimeError) as _exc:
        log("Error: %s" % _exc)
        sys.exit(2)
