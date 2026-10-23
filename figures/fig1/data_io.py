"""Shared workbook loading, validation and figure-output helpers for Fig. 1.

The computational and plotting modules own their scientific calculations; this
module only defines the workbook sheet names, stable palette constants, and
small I/O helpers used by those modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MM_PER_INCH = 25.4
PROBES = (2, 4, 8, 12)
HIGH_COLOR = "#ED1C24"
LOW_COLOR = "#00AEEF"
BULK_COLOR = "#4C78A8"
SCRNA_COLOR = "#F28E2B"
EDGE_COLOR = "#454545"

E_FILL_FRACTIONS = {2: 0.25, 4: 0.50, 8: 0.75, 12: 1.00}
E_MARKER_DIAMETER_PT = 6.0
E_MARKER_OUTLINE_WIDTH_PT = 0.25
E_BASELINE_COLOR = "#777777"
E_GENE_PALETTES = {
    "High": {
        "Gsg1l": "#7F1D3A",
        "Bdnf": "#D23F31",
        "Dkk3": "#B48A00",
        "Mapk4": "#E86E19",
        "Slc17a6": "#8E4B24",
        "Cpne4": "#C24E8B",
        "Gad1": "#B68757",
    },
    "Low": {
        "Opn3": "#2F5AA6",
        "Fibcd1": "#0096A8",
        "Arhgap25": "#2E8B57",
        "Wfs1": "#7853A6",
    },
}
E_GENE_COLORS = {
    gene: color
    for group_colors in E_GENE_PALETTES.values()
    for gene, color in group_colors.items()
}

REGION_COLORS = {
    "OLF": "#43A756",
    "Isocortex": "#EE9999",
    "TH_ex": "#8882BD",
    "TH_RT": "#373486",
    "MB": "#92C981",
    "CB": "#347CB4",
    "STR": "#E34A33",
    "HPF": "#E78BC1",
    "TH_AV": "#C5BBDE",
    "HY": "#F1B46D",
    "HB": "#9B7768",
    "IP": "#79B8D8",
    "fiber": "#D6E4EE",
    "other": "#F1F1F1",
}
CLASS_COLORS = {
    "Main-expression regions": HIGH_COLOR,
    "Reference-negative regions": LOW_COLOR,
    "Other regions": "#DEDEDE",
}

SHEET_B = "fig1_B_counts"
SHEET_C = "fig1_C_values"
SHEET_E = "fig1_E_values"
SHEET_F = "fig1_F_assignment"
SHEET_F_REF = "fig1_F_reference"
SHEET_MODEL = "fig1_F_model"
SHEET_REGIONS = "sfig2_regions"
SHEET_D = "sfig2_D_stats"


def set_style() -> None:
    """Set the publication style shared by the standalone plotting modules."""

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.linewidth": 0.7,
            "axes.labelcolor": "#111111",
            "axes.edgecolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.transparent": True,
        }
    )


def fig_with_axes(
    width_mm: float,
    height_mm: float,
    *,
    left_mm: float,
    right_mm: float,
    bottom_mm: float,
    top_mm: float,
) -> tuple[plt.Figure, plt.Axes]:
    if left_mm + right_mm >= width_mm or bottom_mm + top_mm >= height_mm:
        raise ValueError("Axes margins leave no plotting area")
    fig = plt.figure(figsize=(width_mm / MM_PER_INCH, height_mm / MM_PER_INCH))
    ax = fig.add_axes(
        [
            left_mm / width_mm,
            bottom_mm / height_mm,
            (width_mm - left_mm - right_mm) / width_mm,
            (height_mm - bottom_mm - top_mm) / height_mm,
        ]
    )
    return fig, ax


def save_panel(fig: plt.Figure, panel_dir: Path, stem: str, *, dpi: int = 600) -> None:
    panel_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(panel_dir / f"{stem}.pdf", transparent=True)
    fig.savefig(panel_dir / f"{stem}.svg", transparent=True)
    fig.savefig(panel_dir / f"{stem}.png", dpi=dpi, facecolor="white")
    plt.close(fig)


def save_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")


def read_sheet(workbook: Path, sheet: str) -> pd.DataFrame:
    """Read one required real XLSX sheet; there is no CSV fallback."""

    try:
        frame = pd.read_excel(workbook, sheet_name=sheet)
    except ValueError as exc:
        raise ValueError(f"Workbook {workbook} lacks required sheet {sheet!r}") from exc
    if frame.empty:
        raise ValueError(f"Workbook sheet {sheet!r} is empty")
    return frame


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} lacks required columns: {missing}")


def validate_quantitative_sources(
    counts: pd.DataFrame,
    line: pd.DataFrame,
    quadrant: pd.DataFrame,
    definitions: pd.DataFrame,
) -> None:
    """Validate workbook sheet shapes and source definitions before plotting."""

    require_columns(counts, ("category", "assay", "dataset_count"), SHEET_B)
    require_columns(
        line,
        (
            "gene", "probes", "region_mode", "gene_total_spots",
            "gapdh_total_spots", "gene_gapdh_total_ratio", "n_selected_cells",
            "baseline_ratio", "ratio_vs_num2", "group",
        ),
        SHEET_C,
    )
    require_columns(
        quadrant,
        (
            "gene", "probes", "region_mode", "gene_total_spots",
            "gapdh_total_spots", "gene_gapdh_total_ratio", "n_selected_cells",
            "baseline_ratio", "x_ratio", "group", "false_ratio",
            "baseline_false_ratio", "y_log2", "false_region_total_spots",
            "all_region_total_spots",
        ),
        SHEET_E,
    )
    require_columns(
        definitions,
        ("gene", "region_set", "region", "order", "annotation_column"),
        SHEET_REGIONS,
    )
    if set(definitions["annotation_column"].astype(str)) != {"brain_region"}:
        raise ValueError("SFig. 2 definitions must use the brain_region annotation")
    if set(definitions["region_set"].astype(str)) != {
        "allen_top3_main_expression", "reference_negative",
    }:
        raise ValueError("Unexpected region-set labels in SFig. 2 definitions")
    for name, frame in ((SHEET_C, line), (SHEET_E, quadrant)):
        probes = pd.to_numeric(frame["probes"], errors="raise").astype(int)
        if set(probes) != set(PROBES) or len(frame) != 44:
            raise ValueError(f"{name} must contain exactly 11 genes x 4 probe rows")
        if frame.assign(probes=probes).duplicated(["gene", "probes"]).any():
            raise ValueError(f"{name} contains duplicate gene/probe rows")
        if set(frame["region_mode"].astype(str)) != {"top3"}:
            raise ValueError(f"{name} is not the top3 brain-region result")
