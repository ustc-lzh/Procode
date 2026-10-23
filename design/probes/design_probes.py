#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the four-stage probe design workflow.

The stages select transcripts, screen target regions, filter candidates, and
assemble padlock/readout sequences.  A JSON configuration is optional; the
module defaults and command-line overrides are sufficient for inspection and
small runs.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from sequence_utils import (Config, Resources, ensure_dirs, load_config,      # noqa: E402
                         load_target_region, log, step, sub)

STEPS = ("step1", "step2", "step3", "step4")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="design_probes.py",
        description="BASSFISH probe design and assembly workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Steps: step1 gene detection / step2 target design / step3 probe filtering / step4 assembly; all = all stages")
    parser.add_argument("steps", nargs="+", choices=list(STEPS) + ["all"],
                        help="steps to execute")
    parser.add_argument("--config", default=None,
                        help="optional configuration file (JSON)")

    group = parser.add_argument_group("Common overrides")
    group.add_argument("--project", default=None, help="project name (determines output/<project>/)")
    group.add_argument("--species", default=None, choices=["human", "mouse"])
    group.add_argument("--sample-name", default=None, help="sample prefix in output filenames")
    group.add_argument("--name", default=None, help="batch label")
    group.add_argument("--run-id", default=None,
                       help="run directory identifier; omitted values receive a random suffix")
    group.add_argument("--organ", default=None, help="bulkRNA tissue name, e.g. 'Small Intestine'")
    group.add_argument("--genes", default=None, help="gene names, comma-separated")
    group.add_argument("--gene-file", default=None, help="gene-name list file (one per line)")
    group.add_argument("--work-dir", default=None, help="working directory (root for all intermediates/results)")
    group.add_argument("--seed", type=int, default=None, help="random seed (makes encode selection reproducible)")

    group = parser.add_argument_group("Resource limits")
    group.add_argument("--max-threads", type=int, default=None, help="maximum worker count")
    group.add_argument("--min-free-gb", type=float, default=None, help="minimum free disk space")
    group.add_argument("--seq-cache-mb", type=int, default=None, help="reference sequence cache limit (MB)")
    group.add_argument("--max-candidates", type=int, default=None,
                       help="Step2 per-gene candidate limit; skip genes above it (0 = unlimited)")
    group.add_argument("--lock-file", default=None, help="process-level lock file to prevent resource contention")
    group.add_argument("--dump-config", default=None, help="write the effective configuration to this file and exit")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    overrides = {}
    for key in ("project", "species", "sample_name", "name", "run_id",
                "organ", "work_dir", "seed"):
        value = getattr(args, key.replace("-", "_"), None)
        if value is not None:
            overrides[key] = value
    if args.genes:
        overrides["genes"] = [g.strip() for g in args.genes.split(",") if g.strip()]
    if args.gene_file:
        overrides["gene_file"] = args.gene_file
    resource_over = {}
    for key, value in (("max_threads", args.max_threads), ("min_free_gb", args.min_free_gb),
                       ("seq_cache_mb", args.seq_cache_mb),
                       ("max_candidates", args.max_candidates), ("lock_file", args.lock_file)):
        if value is not None:
            resource_over[key] = value
    if resource_over:
        overrides["resource"] = resource_over

    config_path = args.config if os.path.isfile(args.config) else None
    if args.config and not config_path:
        log("Notice: configuration file not found: %s; using built-in defaults" % args.config)
    cfg = load_config(config_path, overrides)

    if args.dump_config:
        cfg.dump(args.dump_config)
        log("Wrote effective configuration: %s" % args.dump_config)
        return 0

    steps = list(STEPS) if "all" in args.steps else [s for s in STEPS if s in args.steps]

    resumed = os.path.isdir(cfg.run_dir) and bool(os.listdir(cfg.run_dir))

    step("BASSFISH probe design and assembly workflow")
    sub("Config file : %s" % (config_path or "(built-in defaults)"))
    sub("RUN ID   : %s%s" % (cfg.run_id, "   (resuming existing directory)" if resumed else "   (new)"))
    sub("Project/batch: %s / %s" % (cfg.project, cfg.name))
    sub("Species/tissue: %s / %s" % (cfg.species, cfg["organ"]))
    sub("Run directory: %s" % cfg.run_dir)
    sub("Reference directory: %s" % cfg.ref_dir)
    sub("Data directory: %s" % cfg.data_web)
    sub("Target module: %s" % cfg.target_region_dir)
    sub("Steps       : %s" % ", ".join(steps))
    if not resumed and steps and steps[0] != "step1":
        sub("Notice: this run_id has no outputs yet; %s requires results from a previous step; "
            "resume with --run-id set to the earlier run's ID (or --run-id last)" % steps[0])

    ensure_dirs(cfg.work_dir, cfg.run_dir, cfg.report_dir)
    cfg.write_last_run_id()
    res = Resources(cfg)
    res.banner(cfg.work_dir)

    tr = load_target_region(cfg.target_region_dir)
    lock_conf = cfg["resource"]
    run_lock = tr.RunLock(lock_conf.get("lock_file"), float(lock_conf.get("lock_timeout", 1800)))

    t0 = time.time()
    pool = None
    try:
        run_lock.__enter__()
        if "step1" in steps:
            from select_transcripts import run_step1
            run_step1(cfg)

        # ---- Reuse library: use completed genes directly and skip Step2/Step3 -- #
        todo_pairs = None
        cached_pool = None
        cached_genes = None
        if "step2" in steps or "step3" in steps:
            from screen_targets import read_gene_pairs
            from transcript_reference import resolve_from_library
            gene_list_path = os.path.join(cfg.report_dir, "step1_genes.txt")
            if os.path.isfile(gene_list_path):
                gene_pairs = read_gene_pairs(gene_list_path)
                todo_pairs, cached_pool, hit = resolve_from_library(cfg, gene_pairs)
                cached_genes = set(hit)
                if hit:
                    step("Reuse library matched %d genes; skipping Step2/Step3" % len(hit))
                    sub("Using cached results: %s" % ", ".join(hit))
                    sub("New designs needed: %d isoforms" % len(todo_pairs))

        if "step2" in steps and (todo_pairs is None or todo_pairs):
            from screen_targets import run_step2
            run_step2(cfg, res, todo_pairs)
        elif "step2" in steps:
            step("Step2  all genes matched the reuse library; skipping")

        if "step3" in steps and (todo_pairs is None or todo_pairs):
            from filter_targets import run_step3
            pool = run_step3(cfg, res, cached_pool, cached_genes)
        elif "step3" in steps:
            from filter_targets import write_pool_only
            pool = write_pool_only(cfg, cached_pool)

        if "step4" in steps:
            from assemble_probes import run_step4
            run_step4(cfg, res, pool)
    finally:
        run_lock.__exit__()

    step("Workflow complete in %.1f minutes" % ((time.time() - t0) / 60.0))
    sub("RUN ID   : %s" % cfg.run_id)
    sub("Run directory : %s" % cfg.run_dir)
    sub("Output directory: %s" % cfg.output_dir)
    sub("Report directory: %s" % cfg.report_dir)
    sub("Resume this run: python3 design_probes.py --config %s --run-id %s <step>"
        % (os.path.basename(config_path or "(defaults)"), cfg.run_id))
    res.report(cfg.work_dir)
    cfg.dump(os.path.join(cfg.report_dir, "config.used.json"))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("\nInterrupted")
        sys.exit(130)
    except (FileNotFoundError, ValueError, RuntimeError) as _exc:
        log("Error: %s" % _exc)
        sys.exit(2)
