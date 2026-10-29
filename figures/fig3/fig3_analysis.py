"""Run mouse-brain cell-type, expression and brain-region analyses."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

try:
    from .brain_region_analysis import run_correlation, run_correspondence
    from .cell_type_analysis import run_celltypes
    from .data_io import (
        DEFAULT_DATA_ROOT,
        DEFAULT_OUTPUT,
        DEFAULT_WORKBOOK,
        STAGES,
        AnalysisContext,
        DataLocator,
        MissingInput,
        SPATIAL_GENES,
        _jsonable,
        _read_workbook,
        _sheet_rows,
    )
    from .marker_expression import run_markers
    from .spatial_expression import run_spatial
except ImportError:  # direct ``python figures/fig3/fig3_analysis.py``
    from brain_region_analysis import run_correlation, run_correspondence
    from cell_type_analysis import run_celltypes
    from data_io import (
        DEFAULT_DATA_ROOT,
        DEFAULT_OUTPUT,
        DEFAULT_WORKBOOK,
        STAGES,
        AnalysisContext,
        DataLocator,
        MissingInput,
        SPATIAL_GENES,
        _jsonable,
        _read_workbook,
        _sheet_rows,
    )
    from marker_expression import run_markers
    from spatial_expression import run_spatial


def required_inputs(stage: str) -> list[tuple[str, Sequence[str]]]:
    common = [
        ("fig3_refined", ("data/fig3_refined.h5ad", "fig3_refined.h5ad")),
        ("figure_display_annotations", (
            "source_data/figure_display_annotations.csv", "figure_display_annotations.csv",
        )),
    ]
    if stage in {"celltypes", "markers"}:
        return common
    if stage == "correspondence":
        return [
            ("fig3_reference", ("data/fig3_reference.h5ad", "fig3_reference.h5ad")),
            ("adata_cast", ("data/adata_cast.h5ad", "adata_cast.h5ad")),
        ]
    if stage == "correlation":
        return [("correlation_points", (
            "data/tables/D_positive_cell_mean_correlation.csv",
            "tables/D_positive_cell_mean_correlation.csv",
        ))]
    if stage == "spatial":
        return [("spot_counts", (
            "data/spot_cache/counts.json", "spot_cache/counts.json", "counts.json",
        ))] + [
            (f"spot:{gene}", (
                f"data/spot_cache/{gene}.npy", f"spot_cache/{gene}.npy", f"{gene}.npy",
            )) for gene in SPATIAL_GENES
        ]
    return common + required_inputs("correspondence") + required_inputs("correlation") + required_inputs("spatial")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK,
                        help="Figure 3 workbook (XLSX or JSON).")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                        help="Data directory containing data/ and source_data/.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help="Output directory for tables, plots and fig3_report.json.")
    parser.add_argument("--stage", choices=STAGES, default="all",
                        help="Stage: all, celltypes, markers, correspondence, correlation or spatial.")
    parser.add_argument("--check-inputs", action="store_true",
                        help="Print requested input status and exit without analysis.")
    return parser.parse_args(argv)


def check_inputs(args: argparse.Namespace, workbook: dict) -> int:
    locator = DataLocator(args.data_root)
    status = []
    missing = []
    for logical, candidates in required_inputs(args.stage):
        path = locator.find(logical, candidates, required=False)
        status.append({"logical": logical, "found": str(path) if path else None, "required": True})
        if path is None:
            missing.append(logical)

    # Correspondence can use paired H5AD files or a precomputed count table.
    if args.stage in {"all", "correspondence"}:
        ref = locator.find("fig3_reference", ("data/fig3_reference.h5ad", "fig3_reference.h5ad"), required=False)
        cast = locator.find("adata_cast", ("data/adata_cast.h5ad", "adata_cast.h5ad"), required=False)
        counts = locator.find("correspondence_counts", (
            "source_data/F_manual_vs_cast_unmerged_counts.csv",
            "tables/F_manual_vs_cast_unmerged_counts.csv",
        ), required=False)
        if ref is not None and cast is not None or counts is not None or _sheet_rows(workbook, "correspondence_counts"):
            missing = [item for item in missing if item not in {
                "fig3_reference", "adata_cast", "correspondence_counts",
            }]
        else:
            missing.append("correspondence (H5AD pair or precomputed count table)")

    # Correlation points can be read directly from the workbook.
    if args.stage in {"all", "correlation"} and _sheet_rows(workbook, "correlation_points"):
        missing = [item for item in missing if item != "correlation_points"]
    print(json.dumps({"stage": args.stage, "data_root": str(args.data_root),
                      "inputs": status, "missing": missing}, ensure_ascii=False, indent=2))
    return 2 if missing else 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        workbook = _read_workbook(args.workbook)
    except (MissingInput, ValueError, RuntimeError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    if args.check_inputs:
        return check_inputs(args, workbook)

    ctx = AnalysisContext(DataLocator(args.data_root), workbook, args.output_dir)
    try:
        if args.stage in {"all", "celltypes", "markers"}:
            if args.stage == "celltypes":
                run_celltypes(ctx)
            elif args.stage == "markers":
                run_markers(ctx)
            else:
                run_celltypes(ctx)
                run_markers(ctx)
        if args.stage in {"all", "correspondence"}:
            run_correspondence(ctx)
        if args.stage in {"all", "correlation"}:
            run_correlation(ctx)
        if args.stage in {"all", "spatial"}:
            run_spatial(ctx)
    except (MissingInput, FileNotFoundError, ValueError, RuntimeError) as exc:
        ctx.report["error"] = str(exc)
        report_path = ctx.output / "fig3_report.json"
        report_path.write_text(json.dumps(_jsonable(ctx.report), ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"error": str(exc), "report": str(report_path)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    report_path = ctx.output / "fig3_report.json"
    ctx.report["report_json"] = str(report_path)
    report_path.write_text(json.dumps(_jsonable(ctx.report), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(ctx.report), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

