#!/usr/bin/env python3
"""Assign per-gene probe counts from a gene-by-cell-type expression table.

The allocation uses gene-wise maximum expression, whole-reference quantile
anchors, inverse linear mapping to the requested probe-count interval, ceiling,
and clipping.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def assign_probe_number_linear(
    expression: float,
    expression_low: float,
    expression_high: float,
    probe_min: int = 4,
    probe_max: int = 12,
) -> int:
    """Map expression inversely to an integer number of probes."""
    if expression_high == expression_low:
        return probe_max
    value = probe_max - (
        (expression - expression_low) / (expression_high - expression_low)
    ) * (probe_max - probe_min)
    return max(probe_min, min(probe_max, math.ceil(value)))


def read_gene_list(path: Path | None, available: list[str]) -> list[str]:
    if path is None:
        return available
    genes: list[str] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        gene = line.strip().split("\t")[0].split(",")[0].strip()
        if gene and gene.lower() not in {"gene", "gene_name"} and gene not in genes:
            genes.append(gene)
    return genes


def assign_counts(
    expression: pd.DataFrame,
    genes: list[str],
    probe_min: int = 4,
    probe_max: int = 12,
    q_low: float = 0.05,
    q_high: float = 0.95,
    use_log1p: bool = False,
) -> tuple[pd.DataFrame, dict[str, float | int | bool]]:
    if not 0 <= q_low < q_high <= 1:
        raise ValueError("quantiles must satisfy 0 <= q_low < q_high <= 1")
    if probe_min < 1 or probe_max < probe_min:
        raise ValueError("probe counts must satisfy 1 <= probe_min <= probe_max")

    numeric = expression.apply(pd.to_numeric, errors="coerce")
    maximum = numeric.max(axis=1, skipna=True)
    if maximum.index.duplicated().any():
        maximum = maximum.groupby(level=0).max()
    maximum = maximum.dropna().astype(float)
    if maximum.empty:
        raise ValueError("expression table contains no numeric values")
    if use_log1p:
        if (maximum < 0).any():
            raise ValueError("log1p mode requires non-negative expression values")
        maximum = maximum.map(math.log1p)

    expression_low = float(maximum.quantile(q_low))
    expression_high = float(maximum.quantile(q_high))
    rows: list[dict[str, object]] = []
    for gene in genes:
        found = gene in maximum.index
        value = float(maximum.get(gene, 0.0))
        rows.append(
            {
                "gene": gene,
                "max_celltype_expression": value,
                "probe_count": assign_probe_number_linear(
                    value, expression_low, expression_high, probe_min, probe_max
                ),
                "found_in_reference": found,
            }
        )

    result = pd.DataFrame(rows).sort_values(
        ["probe_count", "gene"], ascending=[False, True], ignore_index=True
    )
    metadata: dict[str, float | int | bool] = {
        "q_low": q_low,
        "q_high": q_high,
        "expression_low": expression_low,
        "expression_high": expression_high,
        "probe_min": probe_min,
        "probe_max": probe_max,
        "use_log1p": use_log1p,
        "reference_gene_count": int(maximum.size),
        "requested_gene_count": len(genes),
        "missing_gene_count": int((~result["found_in_reference"]).sum()),
    }
    return result, metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expression", required=True, type=Path)
    parser.add_argument("--genes", type=Path, help="optional gene list; defaults to all rows")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--probe-min", type=int, default=4)
    parser.add_argument("--probe-max", type=int, default=12)
    parser.add_argument("--q-low", type=float, default=0.05)
    parser.add_argument("--q-high", type=float, default=0.95)
    parser.add_argument("--log1p", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    expression = pd.read_csv(args.expression, index_col=0)
    genes = read_gene_list(args.genes, [str(item) for item in expression.index])
    result, metadata = assign_counts(
        expression,
        genes,
        args.probe_min,
        args.probe_max,
        args.q_low,
        args.q_high,
        args.log1p,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output}")
    print(f"wrote {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
