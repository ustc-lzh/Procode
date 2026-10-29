#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Random-access genome intervals and transcript strand annotations."""

from __future__ import annotations

import gzip
import os
import pickle
import time
from typing import Dict, Iterator, List, Optional, Tuple

from sequence_utils import log, sub

__all__ = ["GenomeIndex", "load_gene_bed", "load_transcript_strand",
           "stream_bed", "stream_bed_range", "bed_byte_ranges"]


# --------------------------------------------------------------------------- #
# Random-access genome lookup (faidx style)
# --------------------------------------------------------------------------- #
class GenomeIndex(object):
    """Build per-chromosome indexes of (sequence offset, base count, bases per line, bytes per line).

    FASTA records must have consistent line widths (both hg38.fa and mm10.fa use 50);
    inconsistent widths fall back to "read the whole record + cache" mode for correctness.
    """

    INDEX_VERSION = 1

    def __init__(self, path: str, use_cache: bool = True):
        self.path = os.path.abspath(path)
        if not os.path.isfile(self.path):
            raise FileNotFoundError("Reference genome FASTA not found: %s" % self.path)
        # name -> (seq_offset, n_bases, line_bases, line_bytes, ragged)
        self.records: Dict[str, Tuple[int, int, int, int, bool]] = {}
        self._fh = None
        self._ragged_cache: Dict[str, str] = {}
        self._build(use_cache)

    # -- Build index ------------------------------------------------------- #
    def _cache_path(self) -> str:
        return self.path + ".pgidx"

    def _build(self, use_cache: bool) -> None:
        stat = os.stat(self.path)
        signature = (self.INDEX_VERSION, stat.st_size, int(stat.st_mtime))
        cache_file = self._cache_path()
        if use_cache and os.path.isfile(cache_file):
            try:
                with open(cache_file, "rb") as fh:
                    blob = pickle.load(fh)
                if blob.get("signature") == signature:
                    self.records = blob["records"]
                    sub("Reused genome index cache (%d chromosomes)" % len(self.records))
                    return
            except Exception:
                pass

        t0 = time.time()
        records: Dict[str, Tuple[int, int, int, int, bool]] = {}
        name: Optional[str] = None
        seq_offset = 0
        n_bases = 0
        line_bases = 0
        line_bytes = 0
        ragged = False
        first_line = True
        prev_irregular = False      # Only the last line may be shorter than the line width.

        with open(self.path, "rb") as fh:
            pos = 0
            for raw in fh:
                length = len(raw)
                if raw[:1] == b">":
                    if name is not None:
                        records[name] = (seq_offset, n_bases, line_bases, line_bytes, ragged)
                    name = raw[1:].split()[0].decode("ascii", "replace") if len(raw) > 1 else ""
                    seq_offset = pos + length
                    n_bases = 0
                    line_bases = 0
                    line_bytes = 0
                    ragged = False
                    first_line = True
                    prev_irregular = False
                elif name is not None:
                    bases = len(raw.rstrip(b"\r\n"))
                    if first_line:
                        line_bases, line_bytes = bases, length
                        first_line = False
                    else:
                        if prev_irregular:      # A later line follows a short line -> irregular width.
                            ragged = True
                        prev_irregular = (bases != line_bases or length != line_bytes)
                    n_bases += bases
                pos += length
        if name is not None:
            records[name] = (seq_offset, n_bases, line_bases, line_bytes, ragged)

        self.records = records
        sub("Built genome index: %d chromosomes in %.1f s" % (len(records), time.time() - t0))
        if use_cache:
            try:
                with open(cache_file, "wb") as fh:
                    pickle.dump({"signature": signature, "records": records},
                                fh, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as exc:
                sub("(Could not write genome index cache; this can be ignored: %s)" % exc)

    # -- Fetch sequence ---------------------------------------------------- #
    def __contains__(self, name: str) -> bool:
        return name in self.records

    def length(self, name: str) -> int:
        return self.records[name][1]

    def fetch(self, name: str, start: int, end: int) -> str:
        """Fetch the [start, end) interval (0-based, half-open), equivalent to ``genome[chr][start:end]``."""
        if name not in self.records:
            return ""
        seq_offset, n_bases, line_bases, line_bytes, ragged = self.records[name]
        start = max(0, int(start))
        end = min(int(end), n_bases)
        if end <= start:
            return ""

        if ragged or line_bases <= 0:
            seq = self._ragged_cache.get(name)
            if seq is None:
                seq = self._read_whole(name)
                self._ragged_cache = {name: seq}     # Keep only one record to limit memory growth.
            return seq[start:end].upper()

        newline_bytes = line_bytes - line_bases
        byte_start = seq_offset + start + (start // line_bases) * newline_bytes
        byte_end = seq_offset + end + ((end - 1) // line_bases) * newline_bytes + 1
        if self._fh is None:
            self._fh = open(self.path, "rb")
        self._fh.seek(byte_start)
        raw = self._fh.read(byte_end - byte_start)
        seq = raw.replace(b"\n", b"").replace(b"\r", b"")
        return seq[:end - start].decode("ascii", "replace").upper()

    def _read_whole(self, name: str) -> str:
        seq_offset, n_bases, _lb, _lby, _r = self.records[name]
        if self._fh is None:
            self._fh = open(self.path, "rb")
        self._fh.seek(seq_offset)
        chunks: List[bytes] = []
        got = 0
        while got < n_bases:
            block = self._fh.readline()
            if not block or block[:1] == b">":
                break
            clean = block.rstrip(b"\r\n")
            chunks.append(clean)
            got += len(clean)
        return b"".join(chunks).decode("ascii", "replace").upper()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        self._ragged_cache = {}


# --------------------------------------------------------------------------- #
# Gene interval table (genes.10X.*.all.bed)
# --------------------------------------------------------------------------- #
def load_gene_bed(path: str) -> Dict[str, Tuple[str, int, int]]:
    """gene -> (chr, start, end); retain the first duplicate gene (matching ``.values[0]``)."""
    table: Dict[str, Tuple[str, int, int]] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) < 4:
                continue
            gene = fields[3]
            if gene in table:
                continue
            try:
                table[gene] = (fields[0], int(fields[1]), int(fields[2]))
            except ValueError:
                continue
    return table


# --------------------------------------------------------------------------- #
# Transcript strand
# --------------------------------------------------------------------------- #
def load_transcript_strand(path: str, kind: str, use_cache: bool = True) -> Dict[str, str]:
    """transcript_id (without version) -> '+' / '-'.

    - kind='gtf': stream GENCODE .gtf/.gtf.gz transcript IDs and strand labels.
    - kind='csv': read the transcript table in chunks and retain the needed columns.

    Results are cached in ``<gtf>.strand.pkl`` for reuse.
    """
    cache_file = path + ".strand.pkl"
    if use_cache and os.path.isfile(cache_file):
        try:
            stat = os.stat(path)
            with open(cache_file, "rb") as fh:
                blob = pickle.load(fh)
            if blob.get("signature") == (stat.st_size, int(stat.st_mtime)):
                sub("Reused transcript-strand cache (%d transcripts)" % len(blob["strand"]))
                return blob["strand"]
        except Exception:
            pass

    t0 = time.time()
    if kind == "gtf":
        strand = _strand_from_gtf(path)
    elif kind == "csv":
        strand = _strand_from_csv(path)
    else:
        raise ValueError("load_transcript_strand: kind must be 'gtf' / 'csv'")
    sub("Parsed transcript strands: %d transcripts in %.1f s" % (len(strand), time.time() - t0))

    if use_cache:
        try:
            stat = os.stat(path)
            with open(cache_file, "wb") as fh:
                pickle.dump({"signature": (stat.st_size, int(stat.st_mtime)),
                             "strand": strand}, fh, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            sub("(Could not write strand cache; this can be ignored: %s)" % exc)
    return strand


def _strand_from_gtf(path: str) -> Dict[str, str]:
    opener = gzip.open if path.endswith(".gz") else open
    strand: Dict[str, str] = {}
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:   # type: ignore[operator]
        for line in fh:
            if not line or line[0] == "#":
                continue
            fields = line.split("\t", 9)
            if len(fields) < 9 or fields[2] != "transcript":
                continue
            info = fields[8]
            key = 'transcript_id "'
            pos = info.find(key)
            if pos < 0:
                continue
            pos += len(key)
            end = info.find('"', pos)
            if end < 0:
                continue
            tid = info[pos:end].split(".")[0]
            strand.setdefault(tid, fields[6])
    return strand


def _strand_from_csv(path: str) -> Dict[str, str]:
    import pandas as pd
    strand: Dict[str, str] = {}
    usecols = ["feature", "transcript_id", "strand"]
    for chunk in pd.read_csv(path, usecols=usecols, dtype=str, chunksize=500000):
        chunk = chunk[chunk["feature"] == "transcript"]
        for tid, value in zip(chunk["transcript_id"].values, chunk["strand"].values):
            if isinstance(tid, str):
                strand.setdefault(str(tid).split(".")[0], value)
    return strand


# --------------------------------------------------------------------------- #
# Streaming BED reader (the rbp file has 6.45 million rows and cannot fit in memory whole)
# --------------------------------------------------------------------------- #
def _parse_bed_lines(fh, pos: int = 0,
                     stop: Optional[int] = None) -> Iterator[Tuple[str, int, int]]:
    """Parse a BED file one binary line at a time.

    Track the position by accumulating line lengths: in text mode, ``fh.tell()`` cannot
    be called during ``for line in fh`` (OSError: telling position disabled by next() call).
    """
    for raw in fh:
        pos += len(raw)
        if raw[:1] not in (b"#", b"\n", b"\r", b"") \
                and not raw.startswith(b"track") and not raw.startswith(b"browser"):
            fields = raw.rstrip(b"\r\n").split(b"\t")
            if len(fields) >= 3:
                try:
                    yield (fields[0].decode("ascii", "replace"),
                           int(fields[1]), int(fields[2]))
                except ValueError:
                    pass
        # Stop after reading the line that crosses stop so chunks neither overlap nor gap.
        if stop is not None and pos >= stop:
            return


def stream_bed(path: str) -> Iterator[Tuple[str, int, int]]:
    with open(path, "rb") as fh:
        for item in _parse_bed_lines(fh):
            yield item


def bed_byte_ranges(path: str, parts: int) -> List[Tuple[int, Optional[int]]]:
    """Split a BED file into parts byte ranges for parallel scanning."""
    size = os.path.getsize(path)
    if parts <= 1 or size == 0:
        return [(0, None)]
    step = size // parts
    return [(i * step, None if i == parts - 1 else (i + 1) * step) for i in range(parts)]


def stream_bed_range(path: str, start: int, stop: Optional[int]) -> Iterator[Tuple[str, int, int]]:
    """Scan the BED byte range [start, stop); discard a partial first line and read the crossing tail line."""
    with open(path, "rb") as fh:
        pos = start
        if start:
            fh.seek(start)
            pos += len(fh.readline())     # Discard a possibly truncated partial line.
        for item in _parse_bed_lines(fh, pos, stop):
            yield item
