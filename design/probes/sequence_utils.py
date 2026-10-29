#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared configuration, sequence, FASTA, and resource helpers."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import time
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
_T0 = time.time()


def log(msg: str = "") -> None:
    print(msg, flush=True)


def step(title: str) -> None:
    log("")
    log("=" * 78)
    log("%s   [+%.0fs]" % (title, time.time() - _T0))
    log("=" * 78)


def sub(msg: str) -> None:
    log("  " + msg)


def progress(iterable: Iterable, desc: str = "", total: Optional[int] = None) -> Iterator:
    """Use tqdm when available and a sparse text progress fallback otherwise."""
    try:
        from tqdm import tqdm            # type: ignore
        return iter(tqdm(iterable, desc=desc, total=total))
    except ImportError:
        pass

    items = iterable if hasattr(iterable, "__len__") else list(iterable)
    n = total if total is not None else len(items)          # type: ignore[arg-type]

    def _gen():
        stride = max(1, n // 20)
        for i, item in enumerate(items):                     # type: ignore[arg-type]
            if desc and (i % stride == 0 or i == n - 1):
                log("    %s %d/%d" % (desc, i + 1, n))
            yield item
    return _gen()


# --------------------------------------------------------------------------- #
# Parallel utilities
#
# Parallel stages use at most ``resource.max_threads`` workers.  A single
# worker falls back to serial execution with the same input order.
# --------------------------------------------------------------------------- #
def parallel_map(func, items: Sequence[Any], workers: int, desc: str = "",
                 chunksize: int = 1) -> List[Any]:
    """Parallel map preserving input order. ``func`` must be module-level (pickleable)."""
    items = list(items)
    if workers <= 1 or len(items) <= 1:
        return [func(item) for item in progress(items, desc=desc, total=len(items))]

    from concurrent.futures import ProcessPoolExecutor
    try:
        pool = ProcessPoolExecutor(max_workers=workers)
    except (OSError, ValueError, ImportError) as exc:
        # Fall back only when the process pool cannot start; propagate worker exceptions.
        log("    Process pool creation failed (%s); falling back to serial" % exc)
        return [func(item) for item in progress(items, desc=desc, total=len(items))]

    with pool:
        results: List[Any] = []
        done = 0
        stride = max(1, len(items) // 10)
        for value in pool.map(func, items, chunksize=chunksize):
            results.append(value)
            done += 1
            if desc and (done % stride == 0 or done == len(items)):
                log("    %s %d/%d" % (desc, done, len(items)))
        return results


# --------------------------------------------------------------------------- #
# Target-region resource and BLAST helpers are loaded from the local module.
# --------------------------------------------------------------------------- #
_TR_MODULE = None
_OLIGO_MODULE = None


def load_target_region(target_region_dir: str):
    """Import probe_design.py by path (it adds its directory to sys.path to find oligoprop)."""
    global _TR_MODULE
    if _TR_MODULE is not None:
        return _TR_MODULE
    path = os.path.join(target_region_dir, "probe_design.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "Target probe design module not found: %s\n"
            "Place the target_region directory there, or change target_region_dir in the configuration" % path)
    spec = importlib.util.spec_from_file_location("bassfish_target_region", path)
    module = importlib.util.module_from_spec(spec)                 # type: ignore[arg-type]
    sys.modules["bassfish_target_region"] = module
    spec.loader.exec_module(module)                                 # type: ignore[union-attr]
    _TR_MODULE = module
    return module


def load_oligoprop(target_region_dir: str):
    """Load oligoprop (for padlock secondary-structure checks), shared with target design."""
    global _OLIGO_MODULE
    if _OLIGO_MODULE is not None:
        return _OLIGO_MODULE
    load_target_region(target_region_dir)      # Ensure the directory is already on sys.path.
    import oligoprop                            # type: ignore
    _OLIGO_MODULE = oligoprop
    return oligoprop


# --------------------------------------------------------------------------- #
# Sequence utilities (replacing Bio.Seq / Bio.SeqIO)
# --------------------------------------------------------------------------- #
_COMPLEMENT = str.maketrans("ACGTNacgtnRYKMSWBDHVrykmswbdhv",
                            "TGCANtgcanYRMKSWVHDByrmkswvhdb")


def reverse_complement(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


def make_reverse(_str: str) -> str:
    """Return the reverse complement of a nucleotide sequence."""
    return reverse_complement(str(_str))


def read_fasta(path: str, strip_space: bool = True) -> List[Tuple[str, str]]:
    """Return [(id, sequence), ...].

    ``strip_space=True`` mirrors Bio.SeqIO.FastaIO.SimpleFastaParser:
    spaces in sequences are removed. Padlock sequences are written with a '  ' separator,
    which becomes contiguous when read back; Excel column offsets then use the space-free sequence.
    """
    records: List[Tuple[str, str]] = []
    header: Optional[str] = None
    chunks: List[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(chunks)))
                # Bio uses the part before the first whitespace as the id.
                header = line[1:].split()[0] if line[1:].strip() else ""
                chunks = []
            elif header is not None:
                chunks.append(line.replace(" ", "") if strip_space else line)
    if header is not None:
        records.append((header, "".join(chunks)))
    return records


def write_fasta(path: str, records: Sequence[Tuple[str, str]], wrap: int = 60) -> None:
    """Write FASTA with 60-character lines by default (matching Bio.SeqIO.write(..., 'fasta'))."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for rec_id, seq in records:
            fh.write(">%s\n" % rec_id)
            if not seq:
                fh.write("\n")
                continue
            for i in range(0, len(seq), wrap):
                fh.write(seq[i:i + wrap] + "\n")


# --------------------------------------------------------------------------- #
# General sequence helpers used by the workflow.
# --------------------------------------------------------------------------- #
def longest_common_substring_length(seq1: str, seq2: str) -> int:
    """Length of the longest contiguous common substring (rolling array, O(len1*len2) time, O(len2) space)."""
    if not seq1 or not seq2:
        return 0
    previous = [0] * (len(seq2) + 1)
    max_length = 0
    for char1 in seq1:
        current = [0] * (len(seq2) + 1)
        for j, char2 in enumerate(seq2, start=1):
            if char1 == char2:
                value = previous[j - 1] + 1
                current[j] = value
                if value > max_length:
                    max_length = value
        previous = current
    return max_length


def find_overlap(seq1: str, seq2: str, min_overlap: int = 25) -> bool:
    """Check whether two sequences have an end-to-end overlap >= min_overlap (matching design_share_code)."""
    if len(seq1) > len(seq2):
        seq1, seq2 = seq2, seq1
    len1 = len(seq1)
    for overlap_len in range(min_overlap, len1 + 1):
        if seq1[-overlap_len:] == seq2[:overlap_len]:
            return True
    for overlap_len in range(min_overlap, len1 + 1):
        if seq1[:overlap_len] == seq2[-overlap_len:]:
            return True
    return False


def drop_overlapping_indices(index_seq_pairs: Sequence[Tuple[Any, str]],
                             min_overlap: int) -> set:
    """Drop later sequences that overlap an earlier sequence."""
    drop: set = set()
    n = len(index_seq_pairs)
    for i in range(n):
        idx_i, seq_i = index_seq_pairs[i]
        if idx_i in drop:
            continue
        for j in range(n):
            if i == j:
                continue
            idx_j, seq_j = index_seq_pairs[j]
            if idx_j in drop:
                continue
            if find_overlap(seq_i, seq_j, min_overlap=min_overlap):
                drop.add(idx_j)
    return drop


def del_T_target(df, del_col: str = "target_seq", _type: str = "T"):
    """Remove targets ending in the configured terminal T pattern."""
    if _type == "T":
        mask = df[del_col].astype(str).str.strip().str.endswith("T", na=False)
        return df[~mask]
    if _type == "TT":
        cleaned = (df[del_col].astype(str)
                   .str.replace(r"\s+", "", regex=True)
                   .str.upper())
        mask = cleaned.str.endswith("TT", na=False)
        return df[~mask]
    raise ValueError("del_T_target: _type must be 'T' or 'TT', got %r" % _type)


def select_widest_elements(input_list: Sequence[int], num_elements: int) -> List[int]:
    """Pick num_elements sites with the widest spacing (matching refresh.py)."""
    input_list = list(input_list)
    if num_elements > len(input_list) or num_elements < 1:
        raise ValueError("Number of elements must be between 1 and the length of the input list.")
    if num_elements == 1:
        return [input_list[0]]
    intervals = [input_list[i + 1] - input_list[i] for i in range(1, len(input_list) - 1)]
    widest = sorted(range(1, len(intervals) + 1),
                    key=lambda i: intervals[i - 1], reverse=True)[:num_elements - 2]
    result_indices = [0] + list(widest) + [len(input_list) - 1]
    return [input_list[i] for i in sorted(set(result_indices))]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SPECIES_REF = {
    "human": {
        "genome": "hg38.fa",
        "gene_bed": "genes.10X.hg38.all.bed",
        "rbp_bed": os.path.join("rbp", "human.filter2.bed"),
        "gtf": "gencode.v42.annotation.gtf.gz",
        "gtf_kind": "gtf",
        "blast_db": "hdb",
        "transcriptome": "gencode_v42_transcripts.fasta",
    },
    "mouse": {
        "genome": "mm10.fa",
        "gene_bed": "genes.10X.mm10.all.bed",
        "rbp_bed": os.path.join("rbp", "mouse.filter2.bed"),
        "gtf": "mm10.whole.gtf.csv",
        "gtf_kind": "csv",
        "blast_db": "mdb",
        "transcriptome": "gencode.vM26.transcripts.fasta",
    },
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "project": "probe_redesign_project",
    "species": "human",
    "sample_name": "human",
    "name": "20260817",
    "run_id": None,        # If empty, generate {name}_{8 random characters}; see Config.run_id.
    "organ": "Small Intestine",
    "genes": [],
    "gene_file": None,
    "seed": 0,

    "work_dir": ".",
    "data_web": "./data/web",
    "ref_dir": "./ref",
    "target_region_dir": os.path.join(_HERE, "target_region"),
    "blast_db_dir": "../target_region_matlab",

    "step1": {
        "top_isoform": 3,
        "min_expression": 0.0,
    },
    "step2": {
        "binding_length": 38,
        "fanxiang": 0,
        "isoform_spec": 0,
        "looooong": 5,
        "min_tm": 52.0, "max_tm": 70.0,
        "min_gc": 40.0, "max_gc": 65.0,
        "maxdm": 10, "maxstem": 8,
        "mid_length": 6,
        "singlehomo": 15, "bothhomo": 26,
        "evalue": 1000.0,
        "keep_blast_out": False,
        "skip_existing": True,
    },
    "step3": {
        "overlap_single_probe": 6,
        "overlap_raw_filter": 10,
        "overlap_final": 7,
        "rbp_min_frac": 0.8,
        "apply_rbp_filter": True,
        "blast_align_len": 24,
        "blast_mismatch": 14,
        "blast_evalue": 1000.0,
        "del_T_type": "TT",
        "keep_without_filter": [],
        "blast_refer_id_rule": "gene_name_field",
    },
    "step4": {
        "probe_per_gene": 8,
        # Set counts per gene {gene: n}; if empty, all genes use probe_per_gene.
        # Callers may provide a per-gene allocation table.
        "probe_per_gene_map": {},
        "encode_extra": 30,
        "readout_file": "Readout_seq.txt",
        "primer": "AGGACAGTAGCAGACGTGTCAGCCAGAAAA",
        "encode_used_extra": ["TCCTATTCTCAACCTAACCT"],
        "encode_used_new": [],
        "channel_names": ["488", "561", "647", "750"],
        "select_encode": {
            "ham_dis": 18,
            "edge_long_full": 8,
            "edge_ham_dis": 4,
            "edge_long": 5,
            "max_homopolymer": 4,
            "edge_len": 10,
        },
    },
    "resource": {
        "max_threads": 3,
        "min_free_gb": 2.0,
        "seq_cache_mb": 192,
        "max_candidates": 0,
        "lock_file": None,
        "lock_timeout": 1800.0,
    },
}


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._\-]{1,64}$")
LAST_RUN_FILE = ".last_run_id"


class Config(object):
    """Workflow configuration with JSON overrides and resolved paths.

    Each run has its own ``run_id``; all outputs go under ``{work_dir}/runs/{run_id}/``.
    If no run identifier is supplied, a random suffix is appended to the batch
    label so concurrent runs do not share a directory.
    """

    def __init__(self, data: Dict[str, Any], config_path: Optional[str] = None):
        self.data = data
        self.config_path = config_path
        species = data["species"]
        if species not in SPECIES_REF:
            raise ValueError("species must be human / mouse, got %r" % species)
        self.ref = SPECIES_REF[species]

        base = os.path.dirname(os.path.abspath(config_path)) if config_path else os.getcwd()
        self.work_dir = self._abs(data["work_dir"], base)
        self.data_web = self._abs(data["data_web"], self.work_dir)
        self.ref_dir = self._abs(data["ref_dir"], self.work_dir)
        self.target_region_dir = self._abs(data["target_region_dir"], _HERE)
        self.blast_db_dir = self._abs(data["blast_db_dir"], self.work_dir)
        self._resolve_run_id()

    @staticmethod
    def _abs(path: str, base: str) -> str:
        return path if os.path.isabs(path) else os.path.abspath(os.path.join(base, path))

    # -- run_id ------------------------------------------------------------ #
    def _resolve_run_id(self) -> None:
        import uuid
        run_id = self.data.get("run_id")
        if run_id == "last":
            run_id = self.read_last_run_id()
            if not run_id:
                raise ValueError("--run-id last: %s not found; specify run_id explicitly"
                                 % os.path.join(self.runs_root, LAST_RUN_FILE))
        if not run_id:
            # Concurrent runs on the same day do not collide: date label + 8 random characters.
            run_id = "%s_%s" % (self.data["name"], uuid.uuid4().hex[:8])
        run_id = str(run_id).strip()
        if not _RUN_ID_RE.match(run_id):
            raise ValueError("run_id may contain only letters/digits/dots/underscores/hyphens and be at most 64 characters, "
                             "got %r" % run_id)
        self.data["run_id"] = run_id

    @property
    def run_id(self) -> str:
        return self.data["run_id"]

    @property
    def runs_root(self) -> str:
        return os.path.join(self.work_dir, "runs")

    @property
    def run_dir(self) -> str:
        return os.path.join(self.runs_root, self.run_id)

    def read_last_run_id(self) -> Optional[str]:
        path = os.path.join(self.runs_root, LAST_RUN_FILE)
        if not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read().strip() or None

    def write_last_run_id(self) -> None:
        os.makedirs(self.runs_root, exist_ok=True)
        with open(os.path.join(self.runs_root, LAST_RUN_FILE), "w", encoding="utf-8") as fh:
            fh.write(self.run_id)

    # -- Convenience access ------------------------------------------------ #
    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    @property
    def species(self) -> str:
        return self.data["species"]

    @property
    def sample_name(self) -> str:
        return self.data["sample_name"]

    @property
    def name(self) -> str:
        return str(self.data["name"])

    @property
    def project(self) -> str:
        return self.data["project"]

    # -- run directory layout ------------------------------------------------ #
    def path(self, *parts: str) -> str:
        """Output path for this run."""
        return os.path.join(self.run_dir, *parts)

    def input_path(self, path: str) -> str:
        """Input file path (resolved relative to work_dir, outside the run directory)."""
        return path if os.path.isabs(path) else os.path.join(self.work_dir, path)

    @property
    def redesign_dir(self) -> str:
        return self.path("probe_redesign", self.name)

    @property
    def multi_dir(self) -> str:
        return self.path("probe_redesign", self.name + "_multi_gene_del_copy")

    @property
    def single_dir(self) -> str:
        return self.path("probe_redesign", self.name + "_single")

    @property
    def whole_dir(self) -> str:
        return self.path("probe_redesign", self.name + "_whole")

    @property
    def pool_dir(self) -> str:
        return self.path("data", "target_pool")

    @property
    def output_dir(self) -> str:
        return self.path("output", self.project)

    @property
    def report_dir(self) -> str:
        return self.path("report")

    # -- Reference files --------------------------------------------------- #
    def ref_path(self, key: str) -> str:
        return os.path.join(self.ref_dir, self.ref[key])

    @property
    def bulk_dir(self) -> str:
        return os.path.join(self.data_web, "bulkRNA", self.species)

    def whole_file(self, kind: str, suffix: str) -> str:
        """probe_redesign/{name}_whole/{sample}.{kind}.{name}.{suffix}"""
        return os.path.join(self.whole_dir,
                            "%s.%s.%s.%s" % (self.sample_name, kind, self.name, suffix))

    def dump(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.data, fh, ensure_ascii=False, indent=2)


def load_config(path: Optional[str], overrides: Optional[Dict[str, Any]] = None) -> Config:
    data = json.loads(json.dumps(DEFAULT_CONFIG))     # deep copy
    if path:
        with open(path, "r", encoding="utf-8-sig") as fh:
            _deep_update(data, json.load(fh))
    if overrides:
        _deep_update(data, {k: v for k, v in overrides.items() if v is not None})
    return Config(data, path)


# --------------------------------------------------------------------------- #
# Resource limits used by target-region design and filtering.
# --------------------------------------------------------------------------- #
class Resources(object):
    def __init__(self, cfg: Config):
        tr = load_target_region(cfg.target_region_dir)
        self.tr = tr
        limits = cfg["resource"]
        self.cpu_limit = tr.detect_cpu_limit()
        self.threads = max(1, min(int(limits["max_threads"]), self.cpu_limit))
        # An optional throttle file can lower the worker count between genes.
        self.throttle_file = limits.get("throttle_file") or None
        self.min_free_bytes = int(max(0.0, float(limits["min_free_gb"])) * 1073741824)
        self.seq_cache_mb = int(limits["seq_cache_mb"])
        self.max_candidates = int(limits.get("max_candidates") or 0)
        self.memory_limit_mb = tr.detect_memory_limit_mb()

    def banner(self, work_dir: str) -> None:
        free_gb = self.tr.free_disk_bytes(work_dir) / 1073741824.0
        sub("Resource limits: %d CPUs available (using %d) | memory%s | disk free %.1f GB (minimum %.1f) | sequence cache %d MB"
            % (self.cpu_limit, self.threads,
               (" limit %d MB" % self.memory_limit_mb) if self.memory_limit_mb else "limit not detected",
               free_gb, self.min_free_bytes / 1073741824.0, self.seq_cache_mb))

    def current_threads(self) -> int:
        """Read the thread count dynamically from the throttle file; use the startup value if absent/unreadable."""
        if not self.throttle_file:
            return self.threads
        try:
            with open(self.throttle_file, "r", encoding="utf-8") as fh:
                want = int(json.load(fh).get("threads", self.threads))
        except (OSError, ValueError, TypeError):
            return self.threads
        return max(1, min(want, self.cpu_limit))

    def check_disk(self, path: str, what: str = "") -> None:
        free = self.tr.free_disk_bytes(path)
        if free < self.min_free_bytes:
            raise RuntimeError("Only %.1f GB of disk space is free, below the %.1f GB minimum; aborting%s"
                               % (free / 1073741824.0,
                                  self.min_free_bytes / 1073741824.0,
                                  (" (%s)" % what) if what else ""))

    def report(self, work_dir: str) -> None:
        peak = self.tr.peak_rss_mb()
        sub("Resource usage: peak memory %s | disk still free %.1f GB"
            % ("%.0f MB" % peak if peak is not None else "unknown",
               self.tr.free_disk_bytes(work_dir) / 1073741824.0))


def ensure_dirs(*paths: str) -> None:
    for path in paths:
        os.makedirs(path, exist_ok=True)


_GENE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-@/]*$")


def looks_like_gene_name(name: str) -> bool:
    """Basic gene-name validation: non-empty, no spaces, no unusual symbols."""
    name = str(name).strip()
    return bool(name) and bool(_GENE_NAME_RE.match(name))
