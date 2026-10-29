"""Validate the workbook-backed Fig. 1F allocation inputs.

The workbook contains maximum-over-34-cell reference values. This validator
checks them without creating another input copy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_io import SHEET_F, SHEET_F_REF, SHEET_MODEL, read_sheet, require_columns


def validate(workbook: Path) -> dict[str, object]:
    workbook = Path(workbook).resolve()
    assignment = read_sheet(workbook, SHEET_F)
    reference = read_sheet(workbook, SHEET_F_REF)
    model = read_sheet(workbook, SHEET_MODEL)
    require_columns(assignment, ("gene", "expr_for_probe_number", "expression_group", "probe_num"), SHEET_F)
    require_columns(reference, ("gene", "max_cell_type_mean_expression"), SHEET_F_REF)
    require_columns(model, ("q05", "q95", "probe_min", "probe_max"), SHEET_MODEL)
    if len(model) != 1:
        raise ValueError("Fig. 1F model sheet must contain one row")

    values = pd.to_numeric(reference["max_cell_type_mean_expression"], errors="raise")
    genes = reference["gene"].astype(str)
    duplicate_gene_rows = int(genes.duplicated().sum())
    if duplicate_gene_rows:
        values = values.groupby(genes).max()
    else:
        values.index = genes
    q05 = float(values.quantile(0.05))
    q95 = float(values.quantile(0.95))
    model_row = model.iloc[0]
    if not np.isclose(float(model_row.q05), q05) or not np.isclose(float(model_row.q95), q95):
        raise ValueError(f"Fig. 1F q05/q95 mismatch: workbook {model_row.q05}, {model_row.q95}; reference {q05}, {q95}")
    if int(model_row.probe_min) != 4 or int(model_row.probe_max) != 12:
        raise ValueError("Fig. 1F model range must remain 4-12 probes")

    assigned = assignment.copy()
    assigned["expr_for_probe_number"] = pd.to_numeric(assigned["expr_for_probe_number"], errors="raise")
    assigned["probe_num"] = pd.to_numeric(assigned["probe_num"], errors="raise").astype(int)
    expected_expression = values.reindex(assigned["gene"].astype(str)).to_numpy(dtype=float)
    if np.isnan(expected_expression).any() or not np.allclose(assigned["expr_for_probe_number"], expected_expression):
        raise ValueError("Fig. 1F gene expression values do not match the reference sheet")
    expected_probe = np.floor(np.clip(12 - (assigned["expr_for_probe_number"] - q05) / (q95 - q05) * 8, 4, 12) + 0.5).astype(int)
    if not np.array_equal(expected_probe.to_numpy(), assigned["probe_num"].to_numpy()):
        raise ValueError("Fig. 1F assigned probe numbers fail the half-up model")

    return {
        "workbook": workbook.name,
        "reference_rows": int(len(reference)),
        "unique_gene_rows": int(len(values)),
        "duplicate_gene_rows": duplicate_gene_rows,
        "q05": q05,
        "q95": q95,
        "probe_min": int(model_row.probe_min),
        "probe_max": int(model_row.probe_max),
        "assignment_rows": int(len(assigned)),
        "assignment_verified": True,
        "expression_definition": str(model_row.expression_definition),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workbook",
        type=Path,
        default=Path(__file__).resolve().with_name("fig1_data.xlsx"),
        help="XLSX workbook; default is fig1_data.xlsx beside this script.",
    )
    parser.add_argument("--report", type=Path, default=None, help="Optional validation JSON output path.")
    args = parser.parse_args()
    result = validate(args.workbook)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
