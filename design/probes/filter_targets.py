#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step3: clean targeted probes.

The stage performs the following operations:
  3.1 Scan design results and split each gene into multi / single
  3.2 process_gene / gene_single_function -> per-gene CSVs (add overlap tags, deduplicate)
  3.3 Merge into {sample}.multi/single.{name}.raw.csv
  3.4 Add strand direction -> get_seq_loc maps to the genome -> refresh_anno_df -> *.raw.anno.csv
  3.5 Intersect with the rbp bed (equivalent to bedtools intersect -wa -f 0.8) and remove hits
  3.6 Deduplicate overlaps within each gene -> *.filter.csv
  3.7 Merge multi+single, remove terminal 3' TT -> probe pool
  3.8 Run in-pool blastn-short and remove targets that hit multiple genes
  3.9 Deduplicate overlaps again -> final target-pool CSV

The implementation uses bounded-memory operations without changing the
screening criteria:
  * Use genome_mapping for random genome access instead of loading the full FASTA at once;
  * The original find_location_for_row reread a 1.2 MB bed for every row; this reads it once;
  * Replace bedtools with a pure-Python streaming intersection (the rbp bed has 6.45 million rows,
    so streaming memory is constant); it also runs without bedtools;
  * Stream-parse in-pool blastn through a pipe without writing hundreds of MB of intermediate output;
  * rename columns before removing non-string targets.
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd

from sequence_utils import (Config, Resources, del_T_target, drop_overlapping_indices,
                         ensure_dirs, load_target_region, log, make_reverse,
                         parallel_map, progress, step, sub, write_fasta, read_fasta)
from genome_mapping import (GenomeIndex, bed_byte_ranges, load_gene_bed,
                         load_transcript_strand, stream_bed_range)

__all__ = ["run_step3", "write_pool_only"]

SUFFIX = "_target_region_seq.txt"


def _pool_path(cfg: Config) -> str:
    return os.path.join(cfg.pool_dir, "%s.%s.target_pool.csv" % (cfg.sample_name, cfg.name))


def _write_per_gene_report(cfg: Config, pool: pd.DataFrame) -> pd.DataFrame:
    """Report target counts per gene.

    The table marks three cases for direct use by the assembly stage:
      ok         -- enough targets for step4.probe_per_gene
      too_few    -- targets exist but are insufficient; step4 uses all available
      no_target  -- no targets (for example, the isoform is missing from the transcriptome FASTA
                   or all targets were filtered out)
    """
    ensure_dirs(cfg.report_dir)
    expected = int(cfg["step4"]["probe_per_gene"])
    per_gene_map = {str(k): int(v)
                    for k, v in (cfg["step4"].get("probe_per_gene_map") or {}).items()}
    counts = (pool["gene_name"].value_counts()
              .rename_axis("gene_name").reset_index(name="n_target"))

    # Add genes requested by step1 that have no targets left.
    gene_list_path = os.path.join(cfg.report_dir, "step1_genes.txt")
    if os.path.isfile(gene_list_path):
        requested: List[str] = []
        with open(gene_list_path, "r", encoding="utf-8-sig") as fh:
            for line in fh:
                fields = line.rstrip("\r\n").split("\t")
                if len(fields) >= 2 and fields[0].lower() != "transcript_id":
                    if fields[1] not in requested:
                        requested.append(fields[1])
        missing = [g for g in requested if g not in set(counts["gene_name"])]
        if missing:
            counts = pd.concat(
                [counts, pd.DataFrame({"gene_name": missing, "n_target": 0})],
                axis=0, ignore_index=True)

    # In fitted mode, the expected count differs by gene; look it up row by row.
    counts["n_expected"] = counts["gene_name"].map(
        lambda g: per_gene_map.get(str(g), expected))
    counts["status"] = [
        "ok" if n >= exp else ("no_target" if n == 0 else "too_few")
        for n, exp in zip(counts["n_target"], counts["n_expected"])]
    counts = counts.sort_values(["status", "gene_name"]).reset_index(drop=True)
    counts.to_csv(os.path.join(cfg.report_dir, "step3_target_per_gene.csv"),
                  index=False, encoding="utf-8-sig")

    for status, tip in (("too_few", "fewer targets than required by step4; assembling with the available count"),
                        ("no_target", "no targets produced; the gene will not enter assembly")):
        bad = counts[counts["status"] == status]
        if not bad.empty:
            sub("[%s] %s: %s"
                % (status, tip,
                   ", ".join("%s(%d/%d)" % (r.gene_name, r.n_target, r.n_expected)
                             for r in bad.itertuples())))
    return counts


def write_pool_only(cfg: Config, cached_pool: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Fast path when all genes hit the reuse library: write its results as the probe pool."""
    step("Step3  All genes hit the reuse library; writing the probe pool directly")
    if cached_pool is None or cached_pool.empty:
        raise RuntimeError("No usable results in the reuse library; run Step2/Step3 normally")
    ensure_dirs(cfg.pool_dir, cfg.report_dir)
    path = _pool_path(cfg)
    cached_pool.to_csv(path, index=False)
    sub("Probe pool: %d targets, covering %d genes -> %s"
        % (cached_pool.shape[0], cached_pool["gene_name"].nunique(), path))
    _write_per_gene_report(cfg, cached_pool)
    return cached_pool


# --------------------------------------------------------------------------- #
# 3.1 Scan design results
# --------------------------------------------------------------------------- #
def scan_design_dir(in_dir: str, skip_genes: Optional[Set[str]] = None
                    ) -> Tuple[Dict[str, List[Tuple[str, str]]], List[str], List[str]]:
    """Scan the Step2 output directory.

    Genes in ``skip_genes`` are skipped because their results were taken directly from the reuse
    library; recomputing them would duplicate the library results.
    """
    skip_genes = skip_genes or set()
    gene2iso: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    if not os.path.isdir(in_dir):
        raise FileNotFoundError("Step2 output directory not found: %s" % in_dir)
    for fn in sorted(os.listdir(in_dir)):
        if not fn.endswith(SUFFIX):
            continue
        stem = fn[:-len(SUFFIX)]
        if "_" not in stem:
            sub("[WARN] filename does not match the expected format: %s" % fn)
            continue
        gene, isoform = stem.split("_", 1)
        if gene in skip_genes:
            continue
        gene2iso[gene].append((isoform, os.path.join(in_dir, fn)))

    gene_multi = sorted(g for g, items in gene2iso.items() if len(items) > 1)
    gene_single = sorted(g for g, items in gene2iso.items() if len(items) == 1)
    return dict(gene2iso), gene_multi, gene_single


# --------------------------------------------------------------------------- #
# 3.2 Per-gene CSVs
# --------------------------------------------------------------------------- #
def _read_targets(items: Sequence[Tuple[str, str]]) -> pd.DataFrame:
    frames = []
    for isoform, path in items:
        seqs = [seq for _rid, seq in read_fasta(path)]
        frame = pd.DataFrame(seqs, columns=["target_seq"])
        frame["iso_name"] = isoform
        frames.append(frame)
    df = pd.concat(frames, axis=0) if frames else pd.DataFrame(columns=["target_seq", "iso_name"])
    df["tag"] = "unique"
    df["tag_number"] = 1
    df.index = list(range(df.shape[0]))
    return df


def process_gene(gene: str, items: Sequence[Tuple[str, str]], output_dir: str) -> int:
    """For multi-isoform genes, mark targets shared across isoforms, then deduplicate by sequence."""
    df = _read_targets(items)
    if df.empty:
        return 0
    groups = df.groupby("target_seq")["iso_name"].unique()
    for target_seq, isos in groups.items():
        if len(isos) > 1:
            mask = df["target_seq"] == target_seq
            df.loc[mask, "tag"] = "overlap_" + "_".join(isos)
            df.loc[mask, "tag_number"] = len(isos)
    ensure_dirs(output_dir)
    df_del_copy = df.drop_duplicates(subset=["target_seq"])
    df_del_copy.to_csv(os.path.join(output_dir, gene + "_probe.csv"), header=True, index=True)
    return df_del_copy.shape[0]


def gene_single_function(gene: str, items: Sequence[Tuple[str, str]], output_dir: str,
                         min_overlap: int) -> int:
    """For single-isoform genes, deduplicate directly by overlap."""
    df = _read_targets(items)
    if df.empty:
        return 0
    pairs = list(zip(df.index.tolist(), df["target_seq"].tolist()))
    drop = drop_overlapping_indices(pairs, min_overlap=min_overlap)
    df = df.drop(list(drop), inplace=False)
    ensure_dirs(output_dir)
    df.to_csv(os.path.join(output_dir, gene + "_probe.csv"), header=True, index=True)
    return df.shape[0]


# --------------------------------------------------------------------------- #
# 3.3 Merge
# --------------------------------------------------------------------------- #
def merge_gene_csv(in_dir: str, out_path: str) -> pd.DataFrame:
    frames = []
    if os.path.isdir(in_dir):
        for fn in sorted(os.listdir(in_dir)):
            if not fn.endswith(".csv"):
                continue
            frame = pd.read_csv(os.path.join(in_dir, fn), index_col=0)
            frame["gene_name"] = fn.split("_")[0]
            frames.append(frame)
    if frames:
        df = pd.concat(frames, axis=0)
        df = df[["target_seq", "iso_name", "tag", "tag_number", "gene_name"]]
    else:
        df = pd.DataFrame(columns=["target_seq", "iso_name", "tag", "tag_number", "gene_name"])
    ensure_dirs(os.path.dirname(out_path))
    df.to_csv(out_path, header=True, index=True)
    return df


# --------------------------------------------------------------------------- #
# 3.4 Genome localization
# --------------------------------------------------------------------------- #
def annotate_locations(df: pd.DataFrame, genome: GenomeIndex,
                       gene_bed: Dict[str, Tuple[str, int, int]],
                       desc: str = "locate") -> pd.DataFrame:
    """Equivalent to the original get_seq_loc + refresh_anno_df, with on-demand genome reads."""
    chrs: List[Any] = []
    starts: List[Any] = []
    ends: List[Any] = []
    span_cache: Dict[str, str] = {}
    cache_key: Optional[str] = None

    rows = list(zip(df["gene_name"].tolist(), df["strand"].tolist(), df["target"].tolist()))
    for gene_name, strand, target in progress(rows, desc=desc, total=len(rows)):
        info = gene_bed.get(gene_name)
        if info is None or not isinstance(target, str):
            chrs.append(None); starts.append(None); ends.append(None)
            continue
        chr_id, gene_start, gene_end = info
        if strand == "-":
            sequence = make_reverse(target)
        elif strand == "+":
            sequence = target
        else:
            sub("  Unexpected strand (gene=%s strand=%r); skipping" % (gene_name, strand))
            chrs.append(None); starts.append(None); ends.append(None)
            continue

        key = "%s:%d-%d" % (chr_id, gene_start, gene_end)
        if key != cache_key:
            span_cache = {key: genome.fetch(chr_id, gene_start, gene_end)}
            cache_key = key
        span = span_cache[key]

        pos = span.find(sequence.upper())
        if pos < 0:
            chrs.append(None); starts.append(None); ends.append(None)
        else:
            chrs.append(chr_id)
            starts.append(gene_start + pos)
            ends.append(gene_start + pos + len(sequence))

    out = df.copy()
    out["chr"] = chrs
    out["start"] = starts
    out["end"] = ends
    return out


def refresh_anno_df(df_anno: pd.DataFrame) -> pd.DataFrame:
    """Discard unmatched targets (RNA splice sites spanning exons), cast coordinates to int, matching the original implementation."""
    df_anno = df_anno.fillna(0)
    df_anno = df_anno[~df_anno["chr"].isin([0])]
    df_anno = df_anno.copy()
    df_anno["start"] = df_anno["start"].apply(int)
    df_anno["end"] = df_anno["end"].apply(int)
    return df_anno


# --------------------------------------------------------------------------- #
# 3.5 rbp intersection (pure-Python bedtools intersect -wa -f 0.8)
# --------------------------------------------------------------------------- #
def _intersect_worker(job):
    """Intersect one BED byte-range chunk (module-level for process pooling)."""
    rbp_bed, start, stop, by_chr, min_frac = job
    import bisect
    starts_idx = {c: [item[0] for item in v] for c, v in by_chr.items()}
    max_len = {c: max((item[1] - item[0]) for item in v) if v else 0
               for c, v in by_chr.items()}
    hit: Set[str] = set()
    for b_chr, b_start, b_end in stream_bed_range(rbp_bed, start, stop):
        intervals = by_chr.get(b_chr)
        if not intervals:
            continue
        lo = bisect.bisect_left(starts_idx[b_chr], b_start - max_len[b_chr])
        hi = bisect.bisect_right(starts_idx[b_chr], b_end)
        for idx in range(lo, hi):
            a_start, a_end, target = intervals[idx]
            if target in hit:
                continue
            overlap = min(a_end, b_end) - max(a_start, b_start)
            if overlap > 0 and overlap >= min_frac * (a_end - a_start):
                hit.add(target)
    return hit


def intersect_targets(anno_rows: Sequence[Tuple[str, int, int, str]], rbp_bed: str,
                      min_frac: float = 0.8, workers: int = 1) -> Set[str]:
    """Return target sequences overlapping rbp intervals by >= min_frac.

    Equivalent to ``bedtools intersect -wa -f 0.8 -a anno -b rbp`` taking column 5 (target).
    Approach: index probe intervals (a few thousand) by chromosome, then **stream** the rbp bed
    (6.45 million human rows / 147 MB). Memory depends only on the number of probes, not bed size,
    and bedtools is not required. Split the bed into byte ranges, scan them in parallel, and union
    the results.
    """
    by_chr: Dict[str, List[Tuple[int, int, str]]] = defaultdict(list)
    for chr_id, start, end, target in anno_rows:
        by_chr[chr_id].append((int(start), int(end), target))
    for chr_id in by_chr:
        by_chr[chr_id].sort()
    by_chr = dict(by_chr)

    ranges = bed_byte_ranges(rbp_bed, max(1, workers))
    jobs = [(rbp_bed, start, stop, by_chr, min_frac) for start, stop in ranges]
    parts = parallel_map(_intersect_worker, jobs, workers,
                         desc="rbp intersection (%d chunks)" % len(jobs))
    hit: Set[str] = set()
    for part in parts:
        hit |= part
    sub("  rbp intersection: %d target hits" % len(hit))
    return hit


# --------------------------------------------------------------------------- #
# 3.8 In-pool blast filtering for specificity
# --------------------------------------------------------------------------- #
def _refer_gene(sseqid: str, rule: str) -> str:
    """Extract the gene name from a BLAST subject id.

    A gencode header looks like
      ENST00000347831.9|ENSG00000134545.14|OTTHUMG...|OTTHUMT...|KLRC1-201|KLRC1|1332|protein_coding|
    - gene_name_field (default): take field 6 (index 5), the gene name itself;
    - transcript_name_split: take field 5 and truncate at the first '-'.
      Note that the latter truncates MT-RNR2-201 to MT and H2-K1-201 to H2, incorrectly merging
      gene names containing hyphens.
    """
    fields = sseqid.split("|")
    if rule == "transcript_name_split":
        if len(fields) > 4:
            return fields[4].split("-", 1)[0]
        return sseqid
    if len(fields) > 5:
        return fields[5]
    if len(fields) > 4:
        return fields[4].split("-", 1)[0]
    return sseqid


def blast_multi_gene_targets(lines: Iterable[str], align_len: int, mismatch: int,
                             rule: str) -> Set[str]:
    """Stream BLAST hits and apply the final target filter.

    Return the set of queries that hit >1 gene with alignment_length > align_len and
    mismatches <= mismatch. Recording at most 2 gene names per query is sufficient, so memory
    remains constant.
    """
    genes_per_query: Dict[str, Set[str]] = defaultdict(set)
    flagged: Set[str] = set()
    n_line = 0
    for line in lines:
        line = line.rstrip("\r\n")
        if not line:
            continue
        fields = line.split("\t")
        if len(fields) < 12:
            continue
        n_line += 1
        query = fields[0]
        if query in flagged:
            continue
        try:
            length = int(fields[3])
            mism = int(fields[4])
        except ValueError:
            continue
        if mism > mismatch or length <= align_len:
            continue
        bucket = genes_per_query[query]
        bucket.add(_refer_gene(fields[1], rule))
        if len(bucket) > 1:
            flagged.add(query)
            genes_per_query.pop(query, None)
    sub("  blast results: %d lines; %d targets classified as non-specific" % (n_line, len(flagged)))
    return flagged


# --------------------------------------------------------------------------- #
# Main workflow
# --------------------------------------------------------------------------- #
def _write_bed(df: pd.DataFrame, columns: Sequence[str], path: str) -> None:
    ensure_dirs(os.path.dirname(path))
    df[list(columns)].to_csv(path, sep="\t", header=False, index=False)


def _dedup_worker(job):
    """Deduplicate overlaps for one gene (module-level so process pools can pickle it)."""
    seqs, min_overlap = job
    return drop_overlapping_indices(list(enumerate(seqs)), min_overlap=min_overlap)


def _dedup_by_overlap(df: pd.DataFrame, seq_col: str, min_overlap: int,
                      desc: str, workers: int = 1) -> pd.DataFrame:
    """For overlapping targets within one gene, keep only the first occurrence.

    Genes do not affect one another, so split by gene and run in parallel with a process pool
    (this step performs O(n^2) sequence comparisons and is a major Step3 bottleneck for genes
    with thousands of targets).
    Use **row numbers**, not index labels, to mark drops: at the probe-pool stage the index is the
    target sequence itself, so dropping by label would remove all rows with the same sequence.
    """
    groups = list(df.groupby("gene_name", sort=False))
    if not groups:
        return df.iloc[0:0]
    jobs = [(group[seq_col].tolist(), min_overlap) for _gene, group in groups]
    drops = parallel_map(_dedup_worker, jobs, workers, desc=desc)

    keep_frames = []
    for (_gene, group), drop in zip(groups, drops):
        if drop:
            mask = [i not in drop for i in range(group.shape[0])]
            group = group.iloc[mask]
        keep_frames.append(group)
    return pd.concat(keep_frames, axis=0)


def run_step3(cfg: Config, res: Resources,
              cached_pool: Optional[pd.DataFrame] = None,
              skip_genes: Optional[Set[str]] = None) -> pd.DataFrame:
    """``cached_pool`` / ``skip_genes`` are genes taken directly from the reuse library.
    Do not recompute them; merge them into the probe pool only at the end to avoid duplication."""
    step("Step3  Clean targeted probes")
    params = cfg["step3"]
    tr = load_target_region(cfg.target_region_dir)
    res.check_disk(cfg.work_dir, "Before starting Step3")

    # ---- 3.1 Scan -------------------------------------------------------- #
    if skip_genes is None and cached_pool is not None and not cached_pool.empty:
        skip_genes = set(cached_pool["gene_name"].astype(str))
    gene2iso, gene_multi, gene_single = scan_design_dir(cfg.redesign_dir, skip_genes)
    if skip_genes:
        sub("Genes taken directly from the reuse library and skipped here: %d" % len(skip_genes))
    sub("Total genes: %d  multi-isoform: %d  single-isoform: %d"
        % (len(gene2iso), len(gene_multi), len(gene_single)))
    if not gene2iso:
        raise RuntimeError("Step2 produced no target regions; cannot continue")

    # ---- 3.2 Per-gene CSVs ----------------------------------------------- #
    for gene in progress(gene_multi, desc="multi genes", total=len(gene_multi)):
        process_gene(gene, gene2iso[gene], cfg.multi_dir)
    for gene in progress(gene_single, desc="single genes", total=len(gene_single)):
        gene_single_function(gene, gene2iso[gene], cfg.single_dir,
                             int(params["overlap_single_probe"]))

    # ---- 3.3 Merge ------------------------------------------------------- #
    ensure_dirs(cfg.whole_dir)
    raw_multi_path = cfg.whole_file("multi", "raw.csv")
    raw_single_path = cfg.whole_file("single", "raw.csv")
    df_multi_raw = merge_gene_csv(cfg.multi_dir, raw_multi_path)
    df_single_raw = merge_gene_csv(cfg.single_dir, raw_single_path)
    sub("Raw probes: multi %d / single %d" % (df_multi_raw.shape[0], df_single_raw.shape[0]))

    # ---- 3.4 Strand direction + genome localization --------------------- #
    sub("Reading transcript strand directions: %s" % cfg.ref_path("gtf"))
    strand_map = load_transcript_strand(cfg.ref_path("gtf"), cfg.ref["gtf_kind"])
    sub("  %d transcripts" % len(strand_map))

    gene_bed = load_gene_bed(cfg.ref_path("gene_bed"))
    sub("Gene interval table: %d genes" % len(gene_bed))
    genome = GenomeIndex(cfg.ref_path("genome"))

    anno_frames: Dict[str, pd.DataFrame] = {}
    try:
        for kind, raw_path in (("single", raw_single_path), ("multi", raw_multi_path)):
            df = pd.read_csv(raw_path, index_col=0, header=0)
            if df.empty:
                anno_frames[kind] = df
                continue
            df.index = range(df.shape[0])
            df.columns = ["target", "iso_name", "tag", "tag_number", "gene_name"]
            # Remove non-string targets (blank rows/NaN) before adding strand direction.
            df = df[df["target"].map(type).eq(str)].copy()
            missing = sorted(set(df["iso_name"]) - set(strand_map))
            if missing:
                sub("  [%s] skipped %d isoforms with no strand direction in the GTF: %s"
                    % (kind, len(missing), ", ".join(missing[:5])))
                df = df[df["iso_name"].isin(strand_map)].copy()
            df["strand"] = df["iso_name"].map(strand_map)

            df_anno = annotate_locations(df, genome, gene_bed, desc="[%s] genome localization" % kind)
            df_anno = df_anno[["chr", "start", "end", "gene_name", "target", "iso_name", "strand"]]
            df_anno = refresh_anno_df(df_anno)
            anno_path = cfg.whole_file(kind, "raw.anno.csv")
            _write_bed(df_anno, ["chr", "start", "end", "gene_name", "target", "iso_name", "strand"],
                       anno_path)
            sub("[%s] localized %d / %d -> %s"
                % (kind, df_anno.shape[0], df.shape[0], os.path.basename(anno_path)))
            anno_frames[kind] = df_anno
    finally:
        genome.close()

    # ---- 3.5 rbp intersection: scan the combined single + multi bed once - #
    overlap_targets: Dict[str, Set[str]] = {"single": set(), "multi": set()}
    rbp_bed = cfg.ref_path("rbp_bed")
    all_rows: List[Tuple[str, int, int, str]] = []
    for kind, df_anno in anno_frames.items():
        if not df_anno.empty and "chr" in df_anno.columns:
            all_rows.extend(zip(df_anno["chr"], df_anno["start"],
                                df_anno["end"], df_anno["target"]))
    if params.get("apply_rbp_filter", True) and all_rows:
        if os.path.isfile(rbp_bed):
            sub("rbp intersection (%s, %.0f MB, %d probe intervals, %d processes)"
                % (os.path.basename(rbp_bed), os.path.getsize(rbp_bed) / 1048576.0,
                   len(all_rows), res.current_threads()))
            hits = intersect_targets(all_rows, rbp_bed, float(params["rbp_min_frac"]),
                                     workers=res.current_threads())
            for kind, df_anno in anno_frames.items():
                if df_anno.empty:
                    continue
                kind_hits = set(df_anno[df_anno["target"].isin(hits)]["target"])
                overlap_targets[kind] = kind_hits
                if kind_hits:
                    _write_bed(df_anno[df_anno["target"].isin(kind_hits)],
                               ["chr", "start", "end", "gene_name", "target",
                                "iso_name", "strand"],
                               cfg.whole_file(kind, "raw.anno.overlap.bed"))
        else:
            sub("rbp bed not found (%s); skipping this filter" % rbp_bed)

    # ---- 3.6 rbp filtering + within-gene overlap deduplication ----------- #
    for kind, raw_path in (("single", raw_single_path), ("multi", raw_multi_path)):
        df_raw = pd.read_csv(raw_path)          # Do not set index_col; retain the loci_rank column.
        if df_raw.empty:
            df_raw.to_csv(cfg.whole_file(kind, "filter.csv"), header=True, index=True)
            continue
        hits = overlap_targets.get(kind, set())
        before = df_raw.shape[0]
        if hits:
            df_raw = df_raw[~df_raw["target_seq"].isin(hits)]
        df_drop = _dedup_by_overlap(df_raw, "target_seq",
                                    int(params["overlap_raw_filter"]),
                                    desc="[%s] overlap deduplication" % kind,
                                    workers=res.current_threads())
        out_path = cfg.whole_file(kind, "filter.csv")
        df_drop.to_csv(out_path, header=True, index=True)
        sub("[%s] %d -> %d after rbp filtering -> %d after overlap deduplication"
            % (kind, before, df_raw.shape[0], df_drop.shape[0]))

    # ---- 3.7 Merge probe pool -------------------------------------------- #
    pool_frames = []
    for kind in ("multi", "single"):
        df = pd.read_csv(cfg.whole_file(kind, "filter.csv"), index_col=0)
        if df.empty:
            continue
        df.columns = ["loci_rank", "target", "iso_name", "tag", "tag_number", "gene_name"]
        df.index = df["target"]
        pool_frames.append(df)
    if not pool_frames:
        raise RuntimeError("Step3: no targets remain after filtering; consider relaxing the parameters")
    pool = pd.concat(pool_frames, axis=0)
    pool = pool[["loci_rank", "target", "iso_name", "tag", "tag_number", "gene_name"]]
    sub("Probe pool after merging: %d targets" % pool.shape[0])

    before = pool.shape[0]
    pool = del_T_target(pool, del_col="target", _type=str(params["del_T_type"]))
    sub("Removed terminal 3' %s: %d -> %d" % (params["del_T_type"], before, pool.shape[0]))

    # ---- 3.8 In-pool blastn ---------------------------------------------- #
    ensure_dirs(cfg.pool_dir)
    pool_fasta = os.path.join(cfg.pool_dir, "%s.%s.pool.fasta" % (cfg.sample_name, cfg.name))
    write_fasta(pool_fasta, [(seq, seq) for seq in pool["target"].tolist()])
    sub("Probe pool FASTA: %s" % pool_fasta)

    blastn = tr.find_blast_executable("blastn", cfg.get("blast_bin"))
    db_prefix = tr.resolve_blast_db(cfg.ref["blast_db"], cfg.blast_db_dir)
    sub("In-pool blastn-short (%s, threads=%d, results streamed without writing to disk)"
        % (os.path.basename(db_prefix), res.current_threads()))
    keep_blast = bool(params.get("keep_blast_out", False))
    blast_out = os.path.join(cfg.pool_dir, "%s.%s.pool.blast.txt"
                             % (cfg.sample_name, cfg.name)) if keep_blast else None
    lines = tr.run_blastn(blastn, pool_fasta, db_prefix,
                          float(params["blast_evalue"]), res.current_threads(), "blastn-short",
                          out_path=blast_out, min_free_bytes=res.min_free_bytes)
    flagged = blast_multi_gene_targets(lines, int(params["blast_align_len"]),
                                       int(params["blast_mismatch"]),
                                       str(params.get("blast_refer_id_rule", "gene_name_field")))

    keep_without_filter = set(params.get("keep_without_filter") or [])
    before = pool.shape[0]
    pool = pool.loc[(~pool["target"].isin(flagged))
                    | (pool["gene_name"].isin(keep_without_filter))].copy()
    sub("Non-specific filtering: %d -> %d" % (before, pool.shape[0]))

    # ---- 3.9 Final overlap deduplication --------------------------------- #
    pool = _dedup_by_overlap(pool, "target", int(params["overlap_final"]),
                             desc="final deduplication", workers=res.current_threads())

    # ---- 3.10 Save to reuse library + merge existing library genes -------- #
    from transcript_reference import build_library
    library = build_library(cfg)
    if library.enabled:
        # Isoform level: Step2 products (saves the next blastn run).
        saved_iso = 0
        for gene, items in gene2iso.items():
            for isoform, path in items:
                if not library.has_target_region(gene, isoform):
                    library.save_target_region(gene, isoform, path, cfg.run_id)
                    saved_iso += 1
        if saved_iso:
            sub("Saved target regions for %d isoforms to the reuse library" % saved_iso)

        # Gene level: Step3 products (the next run can skip Step2-3 for the same isoform set).
        saved = 0
        for gene, group in pool.groupby("gene_name", sort=False):
            isoforms = [iso for iso, _p in gene2iso.get(str(gene), [])]
            if not isoforms:
                isoforms = sorted(set(group["iso_name"].astype(str)))
            if not library.has_pool(str(gene), isoforms):
                library.save_pool(str(gene), isoforms, group, cfg.run_id)
                saved += 1
        if saved:
            library.rebuild_index()
            sub("Saved cleaned results for %d genes to the reuse library (the next run can skip Step2-3 with the same parameters and isoforms)"
                % saved)

    if cached_pool is not None and not cached_pool.empty:
        pool = pd.concat([pool, cached_pool], axis=0)
        pool = pool.drop_duplicates(subset=["gene_name", "target"], keep="first")
        sub("Merged %d genes and %d targets taken directly from the reuse library"
            % (cached_pool["gene_name"].nunique(), cached_pool.shape[0]))

    sub("Final probe pool: %d targets, covering %d genes"
        % (pool.shape[0], pool["gene_name"].nunique()))

    final_path = _pool_path(cfg)
    # The index is the target column itself; writing it would collide with the column name
    # (and become target.1 when read back), so omit the index.
    pool.to_csv(final_path, index=False)
    sub("Final probe pool: %s" % final_path)

    _write_per_gene_report(cfg, pool)
    res.report(cfg.work_dir)
    return pool
