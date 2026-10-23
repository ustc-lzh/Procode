"""Run the Figure 2 codebook comparison and Supplementary Figure 3 analyses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from crowding import run_crowding
from data_io import _portable, load_workbook_inputs
from detection_metrics import run_detection
from regional_validation import run_regional


HERE = Path(__file__).resolve().parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workbook",
        type=Path,
        default=HERE / "fig2_data.xlsx",
        help="Current compact Fig. 2 workbook (default: sibling fig2_data.xlsx).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="External directory containing the two required final H5AD files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "results",
        help="Directory for independent source-data tables and panels.",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "crowding", "detection", "regional"),
        default="all",
        help="Analysis stage to run (default: all).",
    )
    parser.add_argument(
        "--check-inputs",
        action="store_true",
        help="List the required workbook/H5AD paths and stop without running analysis.",
    )
    return parser.parse_args(argv)


def _required_paths(args: argparse.Namespace) -> list[Path]:
    required = [args.workbook]
    if args.stage in {"all", "detection", "regional"}:
        if args.data_root is not None:
            root = Path(args.data_root)
            required.extend(
                [
                    root / "adata_good_final_v2_260911.h5ad",
                    root / "adata_bad_final_v2_260911.h5ad",
                ]
            )
        else:
            required.extend(
                [
                    Path("<data-root>") / "adata_good_final_v2_260911.h5ad",
                    Path("<data-root>") / "adata_bad_final_v2_260911.h5ad",
                ]
            )
    return required


def _check_inputs(args: argparse.Namespace) -> None:
    required = _required_paths(args)
    missing = [str(path) for path in required if str(path).startswith("<") or not path.is_file()]
    print(json.dumps({"stage": args.stage, "required": [str(path) for path in required], "missing": missing}, indent=2))
    if missing:
        raise SystemExit(2)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.check_inputs:
        _check_inputs(args)
        return

    try:
        inputs = load_workbook_inputs(args.workbook)
        results: dict[str, Any] = {}
        datasets: dict[str, dict[str, Any]] | None = None
        if args.stage in {"all", "crowding"}:
            results["crowding"] = run_crowding(inputs, args.output_dir)
        if args.stage in {"all", "detection"}:
            results["detection"], datasets = run_detection(
                inputs, args.data_root, args.output_dir, datasets
            )
        if args.stage in {"all", "regional"}:
            results["regional"] = run_regional(
                inputs, args.data_root, args.output_dir, datasets
            )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"analysis.py: error: {exc}") from exc
    print(json.dumps(_portable({"output_dir": str(args.output_dir), "stage": args.stage, "results": results}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
