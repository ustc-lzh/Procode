#!/usr/bin/env python3
"""Run the Fig. 4 analysis and write scientific panels and audit tables.

The entry point uses the retained-cell population and final D1-D5 domain
mapping supplied by the workbook and external data files.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cell_type_maps
import data_io
import domain_composition
import marker_expression
import spatial_domains
import spatial_expression
import supplementary_analysis


def parse_args() -> argparse.Namespace:
    default_workbook = Path(__file__).resolve().with_name("fig4_data.xlsx")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workbook",
        type=Path,
        default=default_workbook,
        help="Fig. 4 workbook; defaults to the sibling fig4_data.xlsx.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Directory or ZIP containing the external data files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fig4_output"),
        help="Directory for analysis tables, figures and the report.",
    )
    parser.add_argument(
        "--stage",
        choices=("logical",),
        default="logical",
        help="Analysis stage. logical writes tables and panels.",
    )
    parser.add_argument("--dpi", type=int, default=400)
    return parser.parse_args()


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> list[str]:
    return supplementary_analysis.save_figure(fig, output_dir, stem, dpi)


def _write_csv(table, output_dir: Path, name: str) -> str | None:
    if table is None:
        return None
    path = output_dir / name
    table.to_csv(path, index=False)
    return path.name


def render_fig4_outputs(
    metadata,
    spots,
    roi,
    config: dict,
    output_dir: Path,
    dpi: int,
) -> list[str]:
    """Write Fig. 4 spatial, UMAP and ROI panels."""

    required_metadata = {
        "display_x_um",
        "display_y_um",
        "final_celltype_code",
        "final_domain",
        "UMAP_1",
        "UMAP_2",
        "in_projected_umap",
        "retained_for_final_figure",
    }
    missing_metadata = sorted(required_metadata.difference(metadata.columns))
    if missing_metadata:
        raise ValueError("Fig. 4 panels require metadata columns: " + ", ".join(missing_metadata) + ".")
    if spots is None:
        raise FileNotFoundError("Transcript-position data are required for Fig. 4.")
    if roi is None or roi.empty:
        raise FileNotFoundError("ROI-center data are required for Fig. 4.")

    retained = metadata.loc[metadata["retained_for_final_figure"]].copy()
    outputs: list[str] = []
    plot_specs = (
        ("Fig4_spatial_celltypes", cell_type_maps.plot_spatial_celltypes, (5.0, 4.4), (retained, config)),
        ("Fig4_projected_umap", cell_type_maps.plot_query_umap, (5.0, 4.4), (metadata, config)),
        ("Fig4_marker_positions", spatial_expression.plot_spots, (5.0, 4.4), (spots, config)),
        ("Fig4_final_domains", spatial_domains.plot_domains, (5.0, 4.4), (retained, config)),
    )
    for stem, plotter, figsize, values in plot_specs:
        figure, ax = plt.subplots(figsize=figsize)
        try:
            plotter(ax, *values)
            outputs.extend(_save_figure(figure, output_dir, stem, dpi))
        except Exception:
            plt.close(figure)
            raise
    for domain in data_io.DOMAIN_ORDER:
        figure, ax = plt.subplots(figsize=(3.6, 3.6))
        try:
            spatial_domains.plot_single_roi(ax, retained, roi, domain, config)
            outputs.extend(_save_figure(figure, output_dir, f"Fig4_roi_{domain}", dpi))
        except Exception:
            plt.close(figure)
            raise
    return outputs


def run(args: argparse.Namespace) -> dict:
    workbook = Path(args.workbook).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    root = args.data_root
    if root is None:
        environment_root = os.environ.get("BIB_DATA_ROOT")
        root = Path(environment_root) if environment_root else None
    locator = data_io.DataLocator(root)
    try:
        sheets = data_io.read_workbook(workbook)
        settings, settings_source = data_io.load_settings(sheets, locator)
        marker_source_table, marker_source = data_io.load_marker_table(sheets, locator)
        config = data_io.prepare_settings(settings, marker_source_table)
        metadata, metadata_source, metadata_checks = data_io.load_metadata(sheets, locator, settings)
        reference, reference_source = data_io.load_reference(sheets, locator)
        counts, counts_source = data_io.load_counts(sheets, locator)
        spots, spots_source = data_io.load_spots(sheets, locator)
        roi, roi_source = data_io.load_roi(sheets, locator)
        roi = data_io.normalize_roi_domains(roi)
        composition_source_table, composition_source = data_io.load_composition_table(sheets, locator)

        marker_table, marker_checks = marker_expression.recompute_markers(
            marker_source_table, counts, metadata, config
        )
        composition, composition_checks = domain_composition.recompute_composition(
            composition_source_table, metadata, config
        )
        marker_file = _write_csv(marker_table, output_dir, "marker_dotplot_analysis.csv")
        composition_file = _write_csv(composition, output_dir, "domain_composition_analysis.csv")
        fig4_files = render_fig4_outputs(metadata, spots, roi, config, output_dir, int(args.dpi))
        sfig6_files = supplementary_analysis.render_sfig6_outputs(
            metadata, reference, marker_table, composition, config, output_dir, int(args.dpi)
        )

        source_registry = {
            "workbook": "fig4_data.xlsx",
            "settings": settings_source,
            "cell_metadata": metadata_source,
            "reference_umap": reference_source,
            "counts": counts_source,
            "transcript_positions": spots_source,
            "roi_centers": roi_source,
            "marker_dotplot": marker_source,
            "domain_composition": composition_source,
        }
        report = {
            "figure": "fig4",
            "sample": str(settings.get("sample", data_io.SAMPLE)),
            "stage": args.stage,
            "workbook": "fig4_data.xlsx",
            "data_root_mode": "zip" if locator.archive is not None else ("directory" if locator.root else "none"),
            "analysis_definition": {
                "retained_population": "retained_for_final_figure",
                "domain_order": data_io.DOMAIN_ORDER,
                "domain_condition": "P(cell type | final domain)",
                "spatial_marker_branch": dict(config["marker_map"]),
                "dotplot_normalization": settings.get(
                    "dotplot_normalization",
                    "log1p(raw_count / max(per-cell total,1) * 34); mean over retained cells within each type; min-max scaling across types within each gene",
                ),
            },
            "sources": source_registry,
            "metadata_checks": metadata_checks,
            "marker_checks": marker_checks,
            "composition_checks": composition_checks,
            "outputs": {
                "fig4": fig4_files,
                "sfig6": sfig6_files,
                "marker_dotplot_analysis": marker_file,
                "domain_composition_analysis": composition_file,
            },
            "judgment": "Coordinate panels and source-backed numerical tables are reproducible from the workbook plus listed external files.",
        }
        (output_dir / "fig4_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return report
    finally:
        locator.close()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

