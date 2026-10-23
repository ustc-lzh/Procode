#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 2: design target regions for the selected transcripts."""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

from sequence_utils import (Config, Resources, ensure_dirs, load_target_region, log, step, sub)

__all__ = ["run_step2", "read_gene_pairs"]


# --------------------------------------------------------------------------- #
# Parallel worker (module-level + process-global; build one transcriptome index per spawned process)
# --------------------------------------------------------------------------- #
_WORKER: Dict[str, object] = {}


def _worker_init(target_region_dir: str, ref_fasta: str, cache_mb: int) -> None:
    from sequence_utils import load_target_region as _load
    tr = _load(target_region_dir)
    _WORKER["tr"] = tr
    _WORKER["index"] = tr.FastaIndex(ref_fasta, cache_mb=cache_mb)


def _worker_design(job):
    """Run one isoform and return a pickleable plain tuple."""
    (gene_name, transcript_id, design_params, paths, blastn,
     threads, poly_pattern, blast_task, keep_blast_out) = job
    tr = _WORKER["tr"]
    index = _WORKER["index"]
    poly_re = re.compile(poly_pattern, re.IGNORECASE)
    try:
        r = tr.design_one(index, gene_name, transcript_id, design_params, paths,   # type: ignore[attr-defined]
                          blastn, threads, False, poly_re, blast_task, keep_blast_out)
        return (gene_name, transcript_id, r.status, r.n_first, r.n_second,
                r.n_final, r.result_file)
    except Exception as exc:                                   # noqa: BLE001
        return (gene_name, transcript_id, "error: %s" % exc, 0, 0, 0, "")


def read_gene_pairs(path: str) -> List[Tuple[str, str]]:
    """Read the two-column (transcript_id, gene_name) list produced by Step1."""
    pairs: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8-sig") as fh:
        for line in fh:
            fields = [f.strip() for f in re.split(r"[\t,]", line.rstrip("\r\n")) if f.strip()]
            if len(fields) < 2:
                continue
            if fields[0].lower() in ("transcript_id", "id"):
                continue
            pairs.append((fields[0], fields[1]))
    return pairs


def run_step2(cfg: Config, res: Resources, gene_pairs: Optional[List[Tuple[str, str]]] = None
              ) -> List[object]:
    step("Step2  target probe design")

    tr = load_target_region(cfg.target_region_dir)
    params = cfg["step2"]

    if gene_pairs is None:
        gene_list_path = os.path.join(cfg.report_dir, "step1_genes.txt")
        if not os.path.isfile(gene_list_path):
            raise FileNotFoundError("Not found: %s; run step1 first (or specify --genes)" % gene_list_path)
        gene_pairs = read_gene_pairs(gene_list_path)
    if not gene_pairs:
        raise ValueError("Step2 has no (transcript_id, gene_name) pairs to design")

    # ---- Reference and database ----------------------------------------- #
    ref_fasta = os.path.join(cfg.blast_db_dir, cfg.ref["transcriptome"])
    db_prefix = tr.resolve_blast_db(cfg.ref["blast_db"], cfg.blast_db_dir)
    blastn = tr.find_blast_executable("blastn", cfg.get("blast_bin"))
    if not tr.check_blast_db(db_prefix):
        raise FileNotFoundError(
            "BLAST database not found: %s(.nin). Put hdb/mdb in blast_db_dir=%s, "
            "or build the database first with target_region --makeblastdb"
            % (db_prefix, cfg.blast_db_dir))

    # ---- Output directory: result goes directly to probe_redesign/{name}/ - #
    tmp_dir = cfg.path("probe_redesign", cfg.name + "_design_tmp")
    ensure_dirs(cfg.redesign_dir, tmp_dir)
    paths = {"out": cfg.path("probe_redesign"), "tmp": tmp_dir,
             "result": cfg.redesign_dir, "db": db_prefix}

    sub("Transcriptome reference: %s" % ref_fasta)
    sub("BLAST database       : %s" % db_prefix)
    sub("Output directory     : %s" % cfg.redesign_dir)
    sub("Isoforms to design   : %d" % len(gene_pairs))
    res.banner(cfg.work_dir)
    res.check_disk(cfg.work_dir, "before Step2 starts")

    design_params: Dict[str, object] = {
        "fanxiang": int(params["fanxiang"]),
        "isoform_spec": int(params["isoform_spec"]),
        "binding_length": int(params["binding_length"]),
        "min_tm": float(params["min_tm"]), "max_tm": float(params["max_tm"]),
        "min_gc": float(params["min_gc"]), "max_gc": float(params["max_gc"]),
        "maxdm": int(params["maxdm"]), "maxstem": int(params["maxstem"]),
        "mid_length": int(params["mid_length"]),
        "singlehomo": int(params["singlehomo"]), "bothhomo": int(params["bothhomo"]),
        "gaplength": int(params.get("gaplength", 2)),
        "apply_gaplength": bool(params.get("apply_gaplength", False)),
        "evalue": float(params["evalue"]),
        "max_candidates": res.max_candidates,
        "min_free_bytes": res.min_free_bytes,
    }
    looooong = int(params["looooong"])
    poly_re = re.compile("|".join(base * looooong for base in "ATCG"), re.IGNORECASE)

    # ---- Resume / reuse library: skip existing results ------------------- #
    from transcript_reference import build_library
    library = build_library(cfg)
    if library.enabled:
        sub("Reuse library: %s" % library.summary())

    skip_existing = bool(params.get("skip_existing", True))
    results: List[Tuple] = []
    todo: List[Tuple[str, str]] = []
    n_resume = n_library = 0
    for transcript_id, gene_name in gene_pairs:
        done_path = os.path.join(cfg.redesign_dir, "%s_%s_target_region_seq.txt"
                                 % (gene_name, transcript_id))
        if skip_existing and os.path.isfile(done_path) and os.path.getsize(done_path) > 0:
            n_resume += 1
        elif library.fetch_target_region(gene_name, transcript_id, cfg.redesign_dir):
            n_library += 1
        else:
            todo.append((transcript_id, gene_name))
            continue
        with open(done_path, "r", encoding="utf-8") as fh:
            n_final = sum(1 for line in fh if line.startswith(">"))
        results.append((gene_name, transcript_id, "ok", 0, 0, n_final, done_path))
    if n_resume:
        sub("Reused results from this run: %d" % n_resume)
    if n_library:
        sub("Matched reuse library (skipped blastn): %d" % n_library)

    # ---- Parallel strategy ---------------------------------------------- #
    # blastn does not scale well with its own threads (3 threads measured only 2.05x, 68% efficiency),
    # so run multiple genes at once with one thread per blastn; total cores still stay within max_threads.
    budget = res.current_threads()
    want = params.get("parallel_genes")
    workers = budget if want in (None, 0, "auto") else int(want)
    workers = max(1, min(workers, budget, len(todo) or 1))
    blast_threads = max(1, budget // workers)
    if todo:
        sub("Parallel strategy: %d genes × %d blastn threads (total %d cores, limit %d)"
            % (workers, blast_threads, workers * blast_threads, budget))

    poly_pattern = poly_re.pattern
    blast_task = params.get("blast_task", "blastn-short")
    keep_blast_out = bool(params.get("keep_blast_out", False))

    if todo and workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        # Run in "waves": reread the thread budget before each wave.
        # A changed throttle value applies at the next batch boundary.
        done = 0
        pool = None
        pool_size = 0
        try:
            index_todo = 0
            while index_todo < len(todo):
                budget_now = res.current_threads()
                want_now = budget_now if want in (None, 0, "auto") else int(want)
                workers_now = max(1, min(want_now, budget_now, len(todo) - index_todo))
                threads_now = max(1, budget_now // workers_now)
                if workers_now != pool_size:
                    if pool is not None:
                        pool.shutdown(wait=True)
                        sub("Parallelism adjusted to %d genes × %d threads (total %d cores)"
                            % (workers_now, threads_now, workers_now * threads_now))
                    pool = ProcessPoolExecutor(
                        max_workers=workers_now, initializer=_worker_init,
                        initargs=(cfg.target_region_dir, ref_fasta, res.seq_cache_mb))
                    pool_size = workers_now

                wave = todo[index_todo:index_todo + workers_now]
                index_todo += len(wave)
                jobs = [(gene_name, transcript_id, design_params, paths, blastn,
                         threads_now, poly_pattern, blast_task, keep_blast_out)
                        for transcript_id, gene_name in wave]
                for row in pool.map(_worker_design, jobs):
                    done += 1
                    sub("[%d/%d] %-14s %-20s %s (target regions: %d)"
                        % (done, len(todo), row[0], row[1], row[2], row[5]))
                    results.append(row)
        finally:
            if pool is not None:
                pool.shutdown(wait=True)
    elif todo:
        sub("Loading transcriptome index ...")
        index = tr.FastaIndex(ref_fasta, cache_mb=res.seq_cache_mb)
        try:
            for n, (transcript_id, gene_name) in enumerate(todo, 1):
                log("")
                sub("[%d/%d] %s  %s" % (n, len(todo), gene_name, transcript_id))
                now = res.current_threads()      # Reread before each gene so throttling takes effect immediately.
                if now != blast_threads:
                    sub("  Thread count adjusted to %d" % now)
                    blast_threads = now
                try:
                    r = tr.design_one(index, gene_name, transcript_id, design_params,
                                      paths, blastn, blast_threads, False, poly_re,
                                      blast_task, keep_blast_out)
                    results.append((gene_name, transcript_id, r.status, r.n_first,
                                    r.n_second, r.n_final, r.result_file))
                except Exception as exc:                   # noqa: BLE001
                    sub("  Error: %s" % exc)
                    results.append((gene_name, transcript_id, "error: %s" % exc, 0, 0, 0, ""))
        finally:
            index.close()

    # ---- Save newly designed regions to the reuse library ---------------- #
    if library.enabled:
        saved = 0
        for row in results:
            gene_name, transcript_id, status = row[0], row[1], row[2]
            if status != "ok":
                continue
            src = os.path.join(cfg.redesign_dir, "%s_%s_target_region_seq.txt"
                               % (gene_name, transcript_id))
            if os.path.isfile(src) and not library.has_target_region(gene_name, transcript_id):
                library.save_target_region(gene_name, transcript_id, src, cfg.run_id)
                saved += 1
        if saved:
            library.rebuild_index()
            sub("Saved to reuse library: %d isoforms (reusable next time with the same parameters)" % saved)

    ok = [r for r in results if r[2] == "ok"]
    log("")
    sub("Step2 complete: %d / %d succeeded" % (len(ok), len(results)))
    for row in results:
        if row[2] != "ok":
            sub("  No output: %-14s %-20s (%s)" % (row[0], row[1], row[2]))

    # Summary table
    ensure_dirs(cfg.report_dir)
    summary_path = os.path.join(cfg.report_dir, "step2_design_summary.csv")
    import csv
    with open(summary_path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["gene_name", "transcript_id", "status", "n_candidate",
                         "n_after_filter", "n_target_region", "result_file"])
        writer.writerows(results)
    sub("Design summary: %s" % summary_path)
    res.report(cfg.work_dir)
    return results
