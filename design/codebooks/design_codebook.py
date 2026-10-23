#!/usr/bin/env python3
"""Design and optimize a gene codebook from single-cell expression priors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from codebook import (
    check_mhd4_codebook,
    codebook_mt_max,
    codebook_round_proportion,
    make_codebook,
    min_pairwise_hamming_distance,
    parallel_simulated_annealing_with_cut_and_viz_proportion,
    select_balanced_subset,
)


def read_genes(path: Path | None, available: list[str]) -> list[str]:
    if path is None:
        return available
    genes: list[str] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        gene = line.strip().split("\t")[0].split(",")[0].strip()
        if gene and gene.lower() not in {"gene", "gene_name"} and gene not in genes:
            genes.append(gene)
    return genes


def load_inputs(
    expression_path: Path,
    abundance_path: Path | None,
    genes_path: Path | None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    gene_by_type = pd.read_csv(expression_path, index_col=0)
    gene_by_type = gene_by_type.apply(pd.to_numeric, errors="coerce")
    if gene_by_type.index.duplicated().any():
        gene_by_type = gene_by_type.groupby(level=0).max()
    genes = read_genes(genes_path, [str(item) for item in gene_by_type.index])
    missing = [gene for gene in genes if gene not in gene_by_type.index]
    if missing:
        raise ValueError(f"genes missing from expression table: {missing[:10]}")
    expression = gene_by_type.loc[genes].T
    if expression.isna().any().any():
        raise ValueError("expression matrix contains missing or non-numeric values")

    if abundance_path is None:
        abundance = pd.Series(1.0, index=expression.index, name="abundance")
    else:
        table = pd.read_csv(abundance_path, index_col=0)
        if table.shape[1] != 1:
            raise ValueError("abundance table must contain exactly one data column")
        abundance = pd.to_numeric(table.iloc[:, 0], errors="raise").reindex(expression.index)
        if abundance.isna().any():
            missing_types = abundance.index[abundance.isna()].tolist()
            raise ValueError(f"abundance missing cell types: {missing_types[:10]}")
    if (abundance < 0).any() or not np.isfinite(abundance.to_numpy()).all():
        raise ValueError("abundance must contain finite non-negative values")
    if float(abundance.sum()) <= 0:
        raise ValueError("abundance sum must be positive")
    return expression, abundance, genes


def bit_names(n_bits: int, channels: int) -> list[str]:
    return [f"R{index // channels + 1}_C{index % channels + 1}" for index in range(n_bits)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expression", required=True, type=Path)
    parser.add_argument("--abundance", type=Path)
    parser.add_argument("--genes", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--code-weight", type=int, default=2)
    parser.add_argument("--min-hamming-distance", type=int, default=2)
    parser.add_argument("--same-channel-round-gap", type=int, default=2)
    parser.add_argument("--max-same-channel-count", type=int, default=2)
    parser.add_argument("--cut-value", type=float, default=700.0)
    parser.add_argument("--num-restarts", type=int, default=3)
    parser.add_argument("--anneal-L", type=int, default=120)
    parser.add_argument("--cooling-q", type=float, default=0.9)
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--skip-optimization", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    expression, abundance, genes = load_inputs(args.expression, args.abundance, args.genes)
    candidates = make_codebook(
        encode_round=args.rounds,
        rounds_per=args.channels,
        same_round_del=args.same_channel_round_gap,
        code_weight=args.code_weight,
        min_hamming_distance=args.min_hamming_distance,
        random_seed=args.random_seed,
        max_same_channel_count=args.max_same_channel_count,
    )
    if len(candidates) < len(genes):
        raise ValueError(
            f"candidate capacity {len(candidates)} is smaller than gene count {len(genes)}"
        )

    initial = select_balanced_subset(
        candidates, len(genes), random_seed=args.random_seed
    )
    initial.index = genes
    initial_score = codebook_round_proportion(initial, expression, abundance)
    initial_max = codebook_mt_max(initial, expression)

    if args.skip_optimization:
        final = initial
        final_score = initial_score
    else:
        final, final_score, _evolution = (
            parallel_simulated_annealing_with_cut_and_viz_proportion(
                codebook=candidates,
                need_num=len(genes),
                final_filtered_exp=expression,
                cell_abundance=abundance,
                cut_value=args.cut_value,
                save_path=None,
                num_restarts=args.num_restarts,
                _L=args.anneal_L,
                _q=args.cooling_q,
                random_seed=args.random_seed,
                use_tqdm=False,
                log_every=args.anneal_L,
            )
        )
        if final is None:
            raise RuntimeError(
                "no solution met the peak-load limit; increase --cut-value"
            )

    final.index = genes
    final_max = codebook_mt_max(final, expression)
    valid = check_mhd4_codebook(
        final,
        rounds_per=args.channels,
        code_weight=args.code_weight,
        min_hamming_distance=args.min_hamming_distance,
        same_round_del=args.same_channel_round_gap,
    )
    minimum_distance = min_pairwise_hamming_distance(final)
    final.columns = bit_names(final.shape[1], args.channels)
    final.index.name = "gene"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    codebook_path = args.output_dir / "codebook.csv"
    metrics_path = args.output_dir / "metrics.json"
    final.to_csv(codebook_path, encoding="utf-8-sig")
    metrics = {
        "n_genes": len(genes),
        "n_cell_types": int(expression.shape[0]),
        "rounds": args.rounds,
        "channels_per_round": args.channels,
        "n_bits": int(final.shape[1]),
        "code_weight": args.code_weight,
        "min_hamming_distance_requested": args.min_hamming_distance,
        "min_hamming_distance_observed": minimum_distance,
        "same_channel_round_gap": args.same_channel_round_gap,
        "max_same_channel_count": args.max_same_channel_count,
        "candidate_capacity": int(len(candidates)),
        "peak_load_limit": args.cut_value if np.isfinite(args.cut_value) else None,
        "initial_proportion_error": initial_score,
        "optimized_proportion_error": float(final_score),
        "relative_error_reduction": (
            float((initial_score - final_score) / initial_score)
            if initial_score > 0
            else 0.0
        ),
        "initial_peak_load": initial_max,
        "optimized_peak_load": final_max,
        "constraints_valid": bool(valid),
        "optimization_skipped": bool(args.skip_optimization),
        "random_seed": args.random_seed,
    }
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {codebook_path}")
    print(f"wrote {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
