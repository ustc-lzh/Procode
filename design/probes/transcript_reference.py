#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reference library for reusing screened target regions.

Store screened target probes grouped by **design binding length** so they can be reused
directly when the same gene is encountered, skipping costly Step2 (blastn) and Step3 (filtering)
and proceeding directly to Step4 assembly.

Directory layout (library_dir defaults to ./library):

    library/{species}/bl{binding_length}/
        target_region/{gene}_{isoform}_target_region_seq.txt   Step2 output (stored by isoform)
        target_region/{gene}_{isoform}.json                    Metadata for this isoform
        pool/{gene}__{isoset}.csv                              This gene's Step3-filtered target rows
        pool/{gene}__{isoset}.json                             Metadata for this gene
        index.csv                                              Summary index for manual lookup

Two reuse levels
--------
* **Isoform level** (target_region): reuse when the design-parameter fingerprint matches, saving Step2 blastn.
  It remains reusable when the tissue or isoform combination changes.
* **Gene level** (pool): skip Step2/Step3 entirely when both the design-parameter fingerprint and isoform combination match.

Fingerprints contain only parameters that affect screening; changing a
parameter therefore prevents incompatible cached results from being reused.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from sequence_utils import Config, ensure_dirs, sub

__all__ = ["TargetLibrary", "design_fingerprint", "clean_fingerprint"]

# Parameters affecting Step2 outputs
DESIGN_KEYS = ("binding_length", "fanxiang", "isoform_spec", "looooong",
               "min_tm", "max_tm", "min_gc", "max_gc", "maxdm", "maxstem",
               "mid_length", "singlehomo", "bothhomo", "evalue")
# Additional parameters affecting Step3 outputs
CLEAN_KEYS = ("overlap_single_probe", "overlap_raw_filter", "overlap_final",
              "rbp_min_frac", "apply_rbp_filter", "blast_align_len",
              "blast_mismatch", "blast_evalue", "del_T_type", "blast_refer_id_rule")

POOL_COLUMNS = ["loci_rank", "target", "iso_name", "tag", "tag_number", "gene_name"]


def _hash(payload: Dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def design_fingerprint(cfg: Config) -> str:
    payload = {"species": cfg.species}
    payload.update({k: cfg["step2"].get(k) for k in DESIGN_KEYS})
    return _hash(payload)


def clean_fingerprint(cfg: Config) -> str:
    payload = {"species": cfg.species, "design": design_fingerprint(cfg)}
    payload.update({k: cfg["step3"].get(k) for k in CLEAN_KEYS})
    return _hash(payload)


def isoset_key(isoforms: Sequence[str]) -> str:
    return _hash({"iso": sorted(str(i) for i in isoforms)})


class TargetLibrary(object):
    def __init__(self, root: str, species: str, binding_length: int,
                 design_fp: str, clean_fp: str, enabled: bool = True):
        # Include parameter fingerprints in the directory so each parameter set has its own results.
        # Comparing fingerprints in metadata alone would also prevent mixing (mismatches trigger a redesign),
        # but later designs would overwrite earlier ones; switching between parameter sets would waste the cache.
        self.root = os.path.join(root, species, "bl%d" % int(binding_length),
                                 "d" + str(design_fp))
        self.species = species
        self.binding_length = int(binding_length)
        self.design_fp = design_fp
        self.clean_fp = clean_fp
        self.enabled = enabled
        self.tr_dir = os.path.join(self.root, "target_region")
        # The probe pool also depends on Step3 parameters, so add another level.
        self.pool_dir = os.path.join(self.root, "c" + str(clean_fp), "pool")
        self.index_path = os.path.join(self.root, "index.csv")
        if enabled:
            ensure_dirs(self.tr_dir, self.pool_dir)

    # -- Isoform level (Step2 output) ------------------------------------- #
    def _tr_paths(self, gene: str, isoform: str) -> Tuple[str, str]:
        stem = "%s_%s" % (gene, isoform)
        return (os.path.join(self.tr_dir, stem + "_target_region_seq.txt"),
                os.path.join(self.tr_dir, stem + ".json"))

    def has_target_region(self, gene: str, isoform: str) -> bool:
        if not self.enabled:
            return False
        fasta, meta = self._tr_paths(gene, isoform)
        if not (os.path.isfile(fasta) and os.path.getsize(fasta) > 0
                and os.path.isfile(meta)):
            return False
        try:
            with open(meta, "r", encoding="utf-8") as fh:
                return json.load(fh).get("design_fp") == self.design_fp
        except Exception:
            return False

    def fetch_target_region(self, gene: str, isoform: str, dest_dir: str) -> bool:
        """Copy a target-region file from the library into this run's directory and return success."""
        fasta, _meta = self._tr_paths(gene, isoform)
        if not self.has_target_region(gene, isoform):
            return False
        ensure_dirs(dest_dir)
        shutil.copy2(fasta, os.path.join(dest_dir, "%s_%s_target_region_seq.txt"
                                         % (gene, isoform)))
        return True

    def save_target_region(self, gene: str, isoform: str, src_path: str,
                           run_id: str) -> None:
        if not self.enabled or not os.path.isfile(src_path):
            return
        fasta, meta = self._tr_paths(gene, isoform)
        ensure_dirs(self.tr_dir)
        shutil.copy2(src_path, fasta)
        with open(src_path, "r", encoding="utf-8") as fh:
            n = sum(1 for line in fh if line.startswith(">"))
        self._write_json(meta, {"gene": gene, "isoform": isoform, "kind": "target_region",
                                "species": self.species,
                                "binding_length": self.binding_length,
                                "design_fp": self.design_fp, "n_target_region": n,
                                "run_id": run_id, "updated_at": _now()})

    # -- Gene level (Step3 output) ---------------------------------------- #
    def _pool_paths(self, gene: str, isoforms: Sequence[str]) -> Tuple[str, str]:
        stem = "%s__%s" % (gene, isoset_key(isoforms))
        return (os.path.join(self.pool_dir, stem + ".csv"),
                os.path.join(self.pool_dir, stem + ".json"))

    def has_pool(self, gene: str, isoforms: Sequence[str]) -> bool:
        if not self.enabled:
            return False
        csv_path, meta = self._pool_paths(gene, isoforms)
        if not (os.path.isfile(csv_path) and os.path.isfile(meta)):
            return False
        try:
            with open(meta, "r", encoding="utf-8") as fh:
                return json.load(fh).get("clean_fp") == self.clean_fp
        except Exception:
            return False

    def load_pool(self, gene: str, isoforms: Sequence[str]) -> pd.DataFrame:
        csv_path, _meta = self._pool_paths(gene, isoforms)
        return pd.read_csv(csv_path)[POOL_COLUMNS]

    def save_pool(self, gene: str, isoforms: Sequence[str], df: pd.DataFrame,
                  run_id: str) -> None:
        if not self.enabled or df.empty:
            return
        csv_path, meta = self._pool_paths(gene, isoforms)
        ensure_dirs(self.pool_dir)
        # Do not write the index: the probe-pool index is the target column itself and would collide with its name.
        df[POOL_COLUMNS].to_csv(csv_path, index=False)
        self._write_json(meta, {"gene": gene, "kind": "pool",
                                "isoforms": sorted(str(i) for i in isoforms),
                                "species": self.species,
                                "binding_length": self.binding_length,
                                "design_fp": self.design_fp, "clean_fp": self.clean_fp,
                                "n_target": int(df.shape[0]),
                                "run_id": run_id, "updated_at": _now()})

    # -- Index ------------------------------------------------------------- #
    @staticmethod
    def _write_json(path: str, payload: Dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

    def rebuild_index(self) -> None:
        if not self.enabled:
            return
        rows: List[Dict[str, Any]] = []
        for folder in (self.tr_dir, self.pool_dir):
            if not os.path.isdir(folder):
                continue
            for fn in sorted(os.listdir(folder)):
                if not fn.endswith(".json"):
                    continue
                try:
                    with open(os.path.join(folder, fn), "r", encoding="utf-8") as fh:
                        meta = json.load(fh)
                except Exception:
                    continue
                rows.append({
                    "kind": meta.get("kind", ""),
                    "gene": meta.get("gene", ""),
                    "isoform": meta.get("isoform", "") or ";".join(meta.get("isoforms", [])),
                    "binding_length": meta.get("binding_length", ""),
                    "n": meta.get("n_target_region", meta.get("n_target", "")),
                    "design_fp": meta.get("design_fp", ""),
                    "clean_fp": meta.get("clean_fp", ""),
                    "run_id": meta.get("run_id", ""),
                    "updated_at": meta.get("updated_at", ""),
                })
        if not rows:
            return
        with open(self.index_path, "w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def summary(self) -> str:
        n_tr = len([f for f in os.listdir(self.tr_dir)
                    if f.endswith("_target_region_seq.txt")]) if os.path.isdir(self.tr_dir) else 0
        n_pool = len([f for f in os.listdir(self.pool_dir)
                      if f.endswith(".csv")]) if os.path.isdir(self.pool_dir) else 0
        return "%s (isoform level: %d, gene level: %d)" % (self.root, n_tr, n_pool)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def resolve_from_library(cfg: Config, gene_pairs: Sequence[Tuple[str, str]]
                         ) -> Tuple[List[Tuple[str, str]], Optional[pd.DataFrame], List[str]]:
    """Look up the reuse library by gene.

    Return ``(list of (transcript_id, gene_name) still needing design, pool table to use directly, matched gene names)``.
    A match requires identical design and cleaning fingerprints and the gene's isoform combination.
    """
    library = build_library(cfg)
    if not library.enabled:
        return list(gene_pairs), None, []

    by_gene: Dict[str, List[str]] = {}
    for transcript_id, gene_name in gene_pairs:
        by_gene.setdefault(gene_name, []).append(transcript_id)

    todo: List[Tuple[str, str]] = []
    frames: List[pd.DataFrame] = []
    hit: List[str] = []
    for gene_name, isoforms in by_gene.items():
        if library.has_pool(gene_name, isoforms):
            try:
                frames.append(library.load_pool(gene_name, isoforms))
                hit.append(gene_name)
                continue
            except Exception as exc:                    # noqa: BLE001
                sub("Failed to read reuse library (%s): %s; redesigning" % (gene_name, exc))
        todo.extend((tid, gene_name) for tid in isoforms)

    cached = pd.concat(frames, axis=0) if frames else None
    return todo, cached, sorted(hit)


def build_library(cfg: Config) -> TargetLibrary:
    lib_conf = cfg.get("library") or {}
    root = lib_conf.get("dir") or "./library"
    root = cfg.input_path(root)
    return TargetLibrary(root, cfg.species, int(cfg["step2"]["binding_length"]),
                         design_fingerprint(cfg), clean_fingerprint(cfg),
                         enabled=bool(lib_conf.get("enabled", True)))
