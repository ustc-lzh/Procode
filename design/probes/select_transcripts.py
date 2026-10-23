#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step1: detect and select isoforms from the input gene_name values.

Each gene is classified into one of four groups:
    ok            ——  present in gtf_basic, with isoforms above the bulkRNA expression threshold; take top-N by expression
    low_expr      ——  present in gtf_basic, but no isoform passes the bulkRNA threshold (too low/filtered expression)
                      still add to the design list, but report separately
    invalid_name  ——  invalid gene-name format (spaces/unusual symbols/empty)
    not_found     ——  valid name but absent from gtf_basic (possibly an alias or typo)

Large annotation and expression tables are read in chunks, retaining only the
columns needed for the selected genes.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from sequence_utils import (Config, ensure_dirs, log, looks_like_gene_name, step, sub)

__all__ = ["run_step1"]

_GTF_USECOLS = ["gene_name", "transcript_id"]


def _load_gene_transcript_map(path: str, genes: Set[str]) -> Dict[str, List[str]]:
    """Keep only target genes as gene_name -> [transcript_id...], reading in chunks to limit memory."""
    mapping: Dict[str, List[str]] = {gene: [] for gene in genes}
    seen: Dict[str, Set[str]] = {gene: set() for gene in genes}
    total = 0
    for chunk in pd.read_csv(path, usecols=_GTF_USECOLS, dtype=str, chunksize=500000):
        total += len(chunk)
        chunk = chunk[chunk["gene_name"].isin(genes)]
        if chunk.empty:
            continue
        for gene, tid in zip(chunk["gene_name"].values, chunk["transcript_id"].values):
            if not isinstance(tid, str):
                continue
            bucket = seen[gene]
            if tid not in bucket:
                bucket.add(tid)
                mapping[gene].append(tid)
    sub("Finished scanning gtf_basic: %d rows, %d genes matched"
        % (total, sum(1 for g in mapping if mapping[g])))
    return mapping


def _load_isoform_mean(path: str, wanted: Set[str]) -> Dict[str, float]:
    """Read the isoform FPKM matrix in chunks, keep needed rows, and incrementally compute row means."""
    means: Dict[str, float] = {}
    for chunk in pd.read_csv(path, index_col=0, chunksize=20000):
        chunk = chunk[chunk.index.isin(wanted)]
        if chunk.empty:
            continue
        numeric = chunk.apply(pd.to_numeric, errors="coerce")
        for tid, value in numeric.mean(axis=1).items():
            means[str(tid)] = float(value)
    return means


def run_step1(cfg: Config) -> pd.DataFrame:
    step("Step1  gene detection + isoform selection")

    params = cfg["step1"]
    top_n = int(params["top_isoform"])
    min_expr = float(params["min_expression"])

    # ---- Input gene list ------------------------------------------------ #
    genes_raw: List[str] = list(cfg.get("genes") or [])
    gene_file = cfg.get("gene_file")
    if gene_file:
        gene_file = cfg.input_path(gene_file)
        with open(gene_file, "r", encoding="utf-8-sig") as fh:
            for line in fh:
                token = line.strip().split("\t")[0].split(",")[0].strip()
                if token and token.lower() not in ("gene", "gene_name"):
                    genes_raw.append(token)
    if not genes_raw:
        raise ValueError("No genes supplied: set genes or gene_file in the configuration")

    # Deduplicate while preserving order.
    genes: List[str] = []
    for gene in genes_raw:
        gene = str(gene).strip()
        if gene and gene not in genes:
            genes.append(gene)
    sub("Input genes: %d" % len(genes))

    # ---- 1) Validate gene names ----------------------------------------- #
    invalid = [g for g in genes if not looks_like_gene_name(g)]
    valid = [g for g in genes if looks_like_gene_name(g)]
    if invalid:
        sub("Invalid gene names (skipped): %s" % ", ".join(invalid))

    # ---- 2) gtf_basic: gene -> isoform --------------------------------- #
    gtf_path = os.path.join(cfg.bulk_dir, "gtf_basic.csv")
    if not os.path.isfile(gtf_path):
        raise FileNotFoundError("gtf_basic.csv not found: %s" % gtf_path)
    sub("Reading %s" % gtf_path)
    gene2iso = _load_gene_transcript_map(gtf_path, set(valid))

    not_found = [g for g in valid if not gene2iso.get(g)]
    if not_found:
        sub("Not found in gtf_basic (possibly an alias or typo): %s" % ", ".join(not_found))
    found = [g for g in valid if gene2iso.get(g)]

    # ---- 3) bulkRNA isoform expression --------------------------------- #
    iso_path = os.path.join(cfg.bulk_dir, "%s_bulkRNA.isoform.csv" % cfg["organ"])
    iso_mean: Dict[str, float] = {}
    if os.path.isfile(iso_path):
        wanted = set()
        for gene in found:
            wanted.update(gene2iso[gene])
        sub("Reading %s (%d candidate isoforms)" % (os.path.basename(iso_path), len(wanted)))
        iso_mean = _load_isoform_mean(iso_path, wanted)
        sub("%d isoforms have records in bulkRNA" % len(iso_mean))
    else:
        sub("Warning: %s not found; all genes treated as low_expr (select isoforms in gtf order)" % iso_path)

    # ---- 4) Classify + select isoforms --------------------------------- #
    rows: List[Dict[str, object]] = []
    selected: List[Tuple[str, str]] = []      # (transcript_id, gene_name)

    for gene in genes:
        if gene in invalid:
            rows.append({"gene_name": gene, "status": "invalid_name", "n_isoform_gtf": 0,
                         "n_isoform_expressed": 0, "selected_isoform": "", "mean_fpkm": ""})
            continue
        if gene in not_found:
            rows.append({"gene_name": gene, "status": "not_found", "n_isoform_gtf": 0,
                         "n_isoform_expressed": 0, "selected_isoform": "", "mean_fpkm": ""})
            continue

        isoforms = gene2iso[gene]
        expressed = [(tid, iso_mean[tid]) for tid in isoforms
                     if tid in iso_mean and iso_mean[tid] > min_expr]
        expressed.sort(key=lambda kv: kv[1], reverse=True)

        if expressed:
            status = "ok"
            picked = expressed[:top_n]
        else:
            # Expression is too low or filtered from bulkRNA: still include it, but report separately.
            status = "low_expr"
            fallback = [(tid, iso_mean.get(tid, 0.0)) for tid in isoforms]
            fallback.sort(key=lambda kv: kv[1], reverse=True)
            picked = fallback[:top_n]

        rows.append({
            "gene_name": gene,
            "status": status,
            "n_isoform_gtf": len(isoforms),
            "n_isoform_expressed": len(expressed),
            "selected_isoform": ";".join(tid for tid, _ in picked),
            "mean_fpkm": ";".join("%.4g" % value for _, value in picked),
        })
        for tid, _value in picked:
            selected.append((tid, gene))

    df = pd.DataFrame(rows, columns=["gene_name", "status", "n_isoform_gtf",
                                     "n_isoform_expressed", "selected_isoform", "mean_fpkm"])

    # ---- 5) Output ------------------------------------------------------ #
    ensure_dirs(cfg.report_dir)
    report_path = os.path.join(cfg.report_dir, "step1_gene_check.csv")
    df.to_csv(report_path, index=False, encoding="utf-8-sig")

    gene_list_path = os.path.join(cfg.report_dir, "step1_genes.txt")
    with open(gene_list_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("transcript_id\tgene_name\n")
        for tid, gene in selected:
            fh.write("%s\t%s\n" % (tid, gene))

    counts = df["status"].value_counts().to_dict()
    log("")
    sub("Classification results: " + " | ".join("%s=%d" % (k, counts.get(k, 0))
                                  for k in ("ok", "low_expr", "not_found", "invalid_name")))
    for status, tip in (("low_expr", "expression too low/filtered from bulkRNA; added to design, review carefully"),
                        ("not_found", "absent from gtf_basic; not included in design"),
                        ("invalid_name", "invalid gene name; not included in design")):
        bad = df[df["status"] == status]["gene_name"].tolist()
        if bad:
            sub("  [%s] %s: %s" % (status, tip, ", ".join(bad)))
    n_multi = int((df[df["status"].isin(["ok", "low_expr"])]["selected_isoform"]
                   .str.count(";") > 0).sum())
    sub("Isoforms to design: %d (%d multi-isoform genes)" % (len(selected), n_multi))
    sub("Detection report: %s" % report_path)
    sub("Design input: %s" % gene_list_path)
    return df
