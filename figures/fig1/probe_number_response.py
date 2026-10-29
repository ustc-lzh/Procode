#!/usr/bin/env python3
"""Generate quantitative Fig. 1 panels B, C and E from the XLSX workbook.

The pie-marker E panel displays the 44-row regional result with the
``log2(false_ratio / baseline_false_ratio)`` sign convention.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Circle, Wedge
from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd

from data_io import (
    E_BASELINE_COLOR,
    E_FILL_FRACTIONS,
    E_GENE_COLORS,
    E_GENE_PALETTES,
    E_MARKER_DIAMETER_PT,
    E_MARKER_OUTLINE_WIDTH_PT,
    HIGH_COLOR,
    LOW_COLOR,
    PROBES,
    SHEET_B,
    SHEET_C,
    SHEET_E,
    SHEET_REGIONS,
    read_sheet,
    set_style,
    validate_quantitative_sources,
)


MM_PER_INCH = 25.4
BULK_COLOR = "#4C78A8"
SCRNA_COLOR = "#F28E2B"

# Fig. 1E uses categorical pie symbols. Every marker has one fixed 6 pt
# diameter; the probe condition is encoded by the filled area fraction.
E_PROBES = PROBES

# Fixed offsets keep labels deterministic while retaining the readable panel
# arrangement.
E_LABEL_OFFSETS = {
    ("Gsg1l", 4): (6, 0, "left", "center", False),
    ("Bdnf", 4): (-6, 0, "right", "center", True),
    ("Dkk3", 4): (-6, 0, "right", "center", True),
    ("Mapk4", 4): (-6, 0, "right", "center", True),
    ("Slc17a6", 4): (0, -6, "center", "top", True),
    ("Cpne4", 4): (6, 0, "left", "center", False),
    ("Gad1", 4): (6, 0, "left", "center", False),
    ("Opn3", 4): (6, 0, "left", "center", False),
    ("Fibcd1", 4): (6, 0, "left", "center", False),
    ("Arhgap25", 4): (6, 0, "left", "center", False),
    ("Wfs1", 4): (6, 0, "left", "center", True),
    ("Gsg1l", 8): (0, -6, "center", "top", True),
    ("Bdnf", 8): (-6, 0, "right", "center", True),
    ("Dkk3", 8): (6, 0, "left", "center", False),
    ("Mapk4", 8): (-6, 0, "right", "center", True),
    ("Slc17a6", 8): (-6, 0, "right", "center", True),
    ("Cpne4", 8): (0, 6, "center", "bottom", True),
    ("Gad1", 8): (6, 0, "left", "center", True),
    ("Opn3", 8): (6, 0, "left", "center", False),
    ("Fibcd1", 8): (6, 0, "left", "center", False),
    ("Arhgap25", 8): (6, 0, "left", "center", False),
    ("Wfs1", 8): (-6, -3.9, "right", "top", True),
    ("Gsg1l", 12): (6, 0, "left", "center", True),
    ("Bdnf", 12): (6, 0, "left", "center", True),
    ("Dkk3", 12): (0, -6, "center", "top", True),
    ("Mapk4", 12): (0, 6, "center", "bottom", True),
    ("Slc17a6", 12): (6, 3.9, "left", "bottom", True),
    ("Cpne4", 12): (-6, 0, "right", "center", True),
    ("Gad1", 12): (6, 0, "left", "center", False),
    ("Opn3", 12): (6, 0, "left", "center", False),
    ("Fibcd1", 12): (6, 0, "left", "center", False),
    ("Arhgap25", 12): (6, 0, "left", "center", False),
    ("Wfs1", 12): (6, 0, "left", "center", False),
}


def _set_style() -> None:
    """Set the shared publication style without changing panel dimensions."""

    set_style()


def _fig_with_axes(
    width_mm: float,
    height_mm: float,
    *,
    left_mm: float,
    right_mm: float,
    bottom_mm: float,
    top_mm: float,
) -> tuple[plt.Figure, plt.Axes]:
    """Create a fixed-size figure and an axes with margins in millimetres."""

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


def _save_panel(fig: plt.Figure, panel_dir: Path, stem: str) -> dict[str, Path]:
    """Save PDF/SVG/PNG without tight cropping or post-save scaling."""

    panel_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "pdf": panel_dir / f"{stem}.pdf",
        "svg": panel_dir / f"{stem}.svg",
        "png": panel_dir / f"{stem}.png",
    }
    fig.savefig(paths["pdf"], transparent=True)
    fig.savefig(paths["svg"], transparent=True)
    fig.savefig(paths["png"], dpi=600, facecolor="white")
    plt.close(fig)
    return paths


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")
    return path


def _validate_workbook(workbook: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the four quantitative XLSX sheets and validate their definitions."""

    counts = read_sheet(workbook, SHEET_B)
    line = read_sheet(workbook, SHEET_C)
    quadrant = read_sheet(workbook, SHEET_E)
    definitions = read_sheet(workbook, SHEET_REGIONS)
    validate_quantitative_sources(counts, line, quadrant, definitions)
    return counts, line, quadrant, definitions


def _plot_panel_b(counts: pd.DataFrame, panel_dir: Path, export_dir: Path) -> dict[str, Any]:
    counts = counts.copy()
    if not {"category", "assay", "dataset_count"}.issubset(counts.columns):
        raise ValueError("Panel B table lacks category, assay or dataset_count")

    # Human normal and Human disease are intentionally pooled here.
    pooled_category = counts["category"].replace(
        {"Human\nnormal": "Human", "Human\ndisease": "Human"}
    )
    merged = (
        counts.assign(category=pooled_category)
        .groupby(["category", "assay"], as_index=False, sort=False)["dataset_count"]
        .sum()
    )
    expected = {
        ("Human", "bulkRNA"): 37,
        ("Human", "scRNA"): 37,
        ("Mouse", "bulkRNA"): 21,
        ("Mouse", "scRNA"): 21,
    }
    observed = {
        (str(row.category), str(row.assay)): int(row.dataset_count)
        for row in merged.itertuples(index=False)
    }
    if observed != expected:
        raise ValueError(f"Unexpected merged panel B counts: {observed}")
    merged["assay_label"] = merged["assay"].map(
        {"bulkRNA": "Bulk RNA-seq", "scRNA": "scRNA-seq"}
    )
    merged = merged[["category", "assay", "assay_label", "dataset_count"]]
    _write_csv(merged, export_dir / "panel_B_merged_counts.csv")

    fig, ax = _fig_with_axes(
        85,
        47,
        left_mm=16,
        right_mm=3,
        bottom_mm=12,
        top_mm=10,
    )
    category_order = ["Human", "Mouse"]
    assay_order = ["bulkRNA", "scRNA"]
    colors = {"bulkRNA": BULK_COLOR, "scRNA": SCRNA_COLOR}
    labels = {"bulkRNA": "Bulk RNA-seq", "scRNA": "scRNA-seq"}
    x = np.arange(len(category_order))
    width = 0.30
    for index, assay in enumerate(assay_order):
        values = (
            merged.loc[merged["assay"] == assay]
            .set_index("category")
            .reindex(category_order)["dataset_count"]
            .to_numpy()
        )
        bars = ax.bar(
            x + (index - 0.5) * width,
            values,
            width=width,
            color=colors[assay],
            edgecolor="none",
            label=labels[assay],
            zorder=3,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.8,
                str(int(value)),
                ha="center",
                va="bottom",
                fontsize=8,
                color="#303030",
            )

    ax.set_ylim(0, 43)
    ax.set_xticks(x, category_order)
    ax.set_ylabel("Number of datasets", fontsize=10)
    ax.yaxis.grid(True, color="#D9D9D9", linewidth=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.2, width=0.7, pad=1.5, labelsize=8)
    fig.legend(
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.56, .895),
        fontsize=8,
        handlelength=1.1,
        columnspacing=0.8,
        borderaxespad=0,
    )
    paths = _save_panel(fig, panel_dir, "panel_B")
    return {"paths": paths, "table": export_dir / "panel_B_merged_counts.csv"}


def _plot_panel_c(line: pd.DataFrame, panel_dir: Path, export_dir: Path) -> dict[str, Any]:
    summary = (
        line.assign(probes=line["probes"].astype(int))
        .groupby(["group", "probes"], as_index=False)["ratio_vs_num2"]
        .agg(mean="mean", lower="min", upper="max")
    )
    summary["region_definition"] = "brain_region_v2 main-expression regions"
    _write_csv(summary, export_dir / "panel_C_summary.csv")

    fig, ax = _fig_with_axes(
        107,
        47,
        left_mm=25,
        right_mm=3,
        bottom_mm=12,
        top_mm=4,
    )
    group_style = {
        "High": (HIGH_COLOR, "High-expression genes"),
        "Low": (LOW_COLOR, "Low-expression genes"),
    }
    for group in ("High", "Low"):
        subset = summary.loc[summary["group"] == group].set_index("probes").reindex([2, 4, 8, 12])
        x = subset.index.to_numpy(dtype=float)
        color, label = group_style[group]
        ax.fill_between(
            x,
            subset["lower"].to_numpy(dtype=float),
            subset["upper"].to_numpy(dtype=float),
            color=color,
            alpha=0.16,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            x,
            subset["mean"].to_numpy(dtype=float),
            "-o",
            color=color,
            lw=1.45,
            ms=3.6,
            markeredgecolor="white",
            markeredgewidth=0.35,
            label=label,
            zorder=3,
        )

    ax.axhline(1, color="#777777", linestyle="--", lw=0.7, alpha=0.65, zorder=1)
    ax.set_xlim(1.6, 12.4)
    ax.set_ylim(0.5, 20.5)
    ax.set_xticks([2, 4, 8, 12])
    ax.set_xlabel("Probe number", fontsize=10)
    ax.set_ylabel("Normalized signal gain\n(fold vs. 2 probes)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.2, width=0.7, pad=1.5, labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="upper left", handlelength=1.5)
    paths = _save_panel(fig, panel_dir, "panel_C")
    return {"paths": paths, "table": export_dir / "panel_C_summary.csv"}


class _ProbeLegendKey:
    """Handle object for the four categorical probe-number symbols."""

    def __init__(self, probes: int):
        self.probes = int(probes)


class _ProbeLegendHandler(HandlerBase):
    def create_artists(
        self,
        legend,
        handle,
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        trans,
    ):
        radius = min(E_MARKER_DIAMETER_PT, height * 0.85) / 2
        center = (width / 2 - xdescent, height / 2 - ydescent)
        fraction = E_FILL_FRACTIONS[handle.probes]
        if fraction == 1:
            fill = Circle(
                center,
                radius,
                facecolor=E_BASELINE_COLOR,
                edgecolor="none",
                transform=trans,
            )
        else:
            fill = Wedge(
                center,
                radius,
                90 - 360 * fraction,
                90,
                facecolor=E_BASELINE_COLOR,
                edgecolor="none",
                transform=trans,
            )
        outline = Circle(
            center,
            radius,
            facecolor="none",
            edgecolor=E_BASELINE_COLOR,
            linewidth=E_MARKER_OUTLINE_WIDTH_PT,
            transform=trans,
        )
        return [fill, outline]


def _pie_marker(probes: int):
    """Return a centred, fixed-diameter path for one categorical pie."""

    fraction = E_FILL_FRACTIONS[int(probes)]
    if fraction == 1:
        return "o"
    # Path.wedge is centred at the first (origin) vertex.  Scaling its unit
    # radius to 0.5 gives the same one-unit diameter as Matplotlib's circle
    # marker, so ``s=6**2`` means 6 pt for every condition.
    path = MplPath.wedge(
        90 - 360 * fraction,
        90,
        n=max(2, int(round(fraction * 8))),
    )
    return MplPath(path.vertices * 0.5, path.codes)


def _draw_e_pie(ax, x: float, y: float, probes: int, color: str, *, zorder: float = 4):
    """Draw one 6 pt pie marker with a true 1/4, 1/2, 3/4 or full fill."""

    probes = int(probes)
    if probes not in E_PROBES:
        raise ValueError(f"Unexpected probe count for Fig. 1E: {probes}")
    marker = _pie_marker(probes)
    ax.scatter(
        [x],
        [y],
        marker=marker,
        s=E_MARKER_DIAMETER_PT**2,
        facecolors=color,
        edgecolors="none",
        linewidths=0,
        zorder=zorder,
    )
    # Draw the thin outline separately so the filled sector cannot erase it.
    ax.scatter(
        [x],
        [y],
        marker="o",
        s=E_MARKER_DIAMETER_PT**2,
        facecolors="none",
        edgecolors=color,
        linewidths=E_MARKER_OUTLINE_WIDTH_PT,
        zorder=zorder + 0.01,
    )


def _add_e_legends(ax):
    """Add gene-colour legends and the categorical probe-number key."""

    gene_font = FontProperties(family="Arial", style="italic", size=8)
    for group, anchor in (("High", (0.58, 0.98)), ("Low", (0.81, 0.98))):
        handles = [
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markerfacecolor=color,
                markeredgecolor="none",
                markersize=4.5,
                label=gene,
            )
            for gene, color in E_GENE_PALETTES[group].items()
        ]
        legend = ax.legend(
            handles=handles,
            title=f"{group}-expression",
            loc="upper left",
            bbox_to_anchor=anchor,
            borderaxespad=0,
            prop=gene_font,
            title_fontsize=9,
            handletextpad=0.4,
            handlelength=0.9,
            labelspacing=0.55,
            frameon=False,
        )
        ax.add_artist(legend)

    ax.legend(
        handles=[_ProbeLegendKey(probe) for probe in E_PROBES],
        labels=[str(probe) for probe in E_PROBES],
        handler_map={_ProbeLegendKey: _ProbeLegendHandler()},
        title="Probe number",
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        borderaxespad=0,
        ncol=4,
        fontsize=8,
        title_fontsize=9,
        handlelength=1.05,
        handletextpad=0.35,
        columnspacing=0.75,
        frameon=False,
    )


def _plot_panel_e(quadrant: pd.DataFrame, panel_dir: Path, export_dir: Path) -> dict[str, Any]:
    """Plot the standalone pie-symbol version of Fig. 1E."""

    data = quadrant.copy()
    data["probes"] = data["probes"].astype(int)
    required = {
        "gene",
        "probes",
        "gene_total_spots",
        "gapdh_total_spots",
        "baseline_ratio",
        "group",
        "false_ratio",
        "baseline_false_ratio",
        "x_ratio",
        "y_log2",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Fig. 1E input lacks required columns: {missing}")
    if len(data) != 44 or data.duplicated(["gene", "probes"]).any():
        raise ValueError("Fig. 1E requires the unchanged 11 genes x 4 probe rows")
    if set(data["probes"]) != set(E_PROBES):
        raise ValueError("Fig. 1E input is missing one of the 2/4/8/12 probe conditions")
    if set(data["gene"]) != set(E_GENE_COLORS):
        raise ValueError("Fig. 1E gene set does not match the bundled palette")

    expected_x = (
        data["gene_total_spots"].astype(float)
        / data["gapdh_total_spots"].astype(float)
        / data["baseline_ratio"].astype(float)
    )
    if not np.allclose(data["x_ratio"].astype(float), expected_x):
        raise ValueError("Fig. 1E x_ratio is not the main-expression gene/Gapdh ratio vs. 2 probes")
    data["y_display"] = np.log2(
        data["false_ratio"].astype(float) / data["baseline_false_ratio"].astype(float)
    )
    if not np.allclose(data["y_display"], -data["y_log2"].astype(float)):
        raise ValueError("Fig. 1E y values do not match log2(false_ratio / baseline_false_ratio)")
    if not np.allclose(data.loc[data["probes"] == 2, "x_ratio"], 1.0) or not np.allclose(
        data.loc[data["probes"] == 2, "y_display"], 0.0
    ):
        raise ValueError("The 11 two-probe baseline rows must share (1, 0)")
    if len(data.loc[data["probes"] != 2]) != 33:
        raise ValueError("Fig. 1E requires 33 non-baseline points")
    for group, genes in E_GENE_PALETTES.items():
        observed = set(data.loc[data["group"] == group, "gene"])
        if observed != set(genes):
            raise ValueError(f"Unexpected {group} gene assignment in Fig. 1E input")

    data["gene_color"] = data["gene"].map(E_GENE_COLORS)
    data["fill_fraction"] = data["probes"].map(E_FILL_FRACTIONS)
    data["marker_diameter_pt"] = E_MARKER_DIAMETER_PT
    data["marker_outline_width_pt"] = E_MARKER_OUTLINE_WIDTH_PT
    data["baseline_drawn_once"] = data["probes"].eq(2)
    _write_csv(data, export_dir / "panel_E_points.csv")

    fig, ax = _fig_with_axes(
        185,
        135,
        left_mm=23,
        right_mm=4,
        bottom_mm=18,
        top_mm=5,
    )
    points = data.loc[data["probes"] != 2].reset_index(drop=True)
    for row in points.itertuples(index=False):
        _draw_e_pie(
            ax,
            float(row.x_ratio),
            float(row.y_display),
            int(row.probes),
            str(row.gene_color),
            zorder=4,
        )
    # Draw the 11 baseline rows once as a shared gray quarter-pie at (1, 0).
    _draw_e_pie(ax, 1.0, 0.0, 2, E_BASELINE_COLOR, zorder=4.1)

    for row in points.itertuples(index=False):
        dx, dy, ha, va, short_leader = E_LABEL_OFFSETS[(str(row.gene), int(row.probes))]
        label_color = tuple(0.80 * np.asarray(mpl.colors.to_rgb(row.gene_color)))
        annotation = {
            "text": str(row.gene),
            "xy": (float(row.x_ratio), float(row.y_display)),
            "xytext": (dx, dy),
            "textcoords": "offset points",
            "ha": ha,
            "va": va,
            "fontsize": 6,
            "fontstyle": "italic",
            "color": label_color,
            "zorder": 6,
        }
        if short_leader:
            annotation["arrowprops"] = {
                "arrowstyle": "-",
                "color": "#666666",
                "lw": 0.35,
                "shrinkA": 1,
                "shrinkB": E_MARKER_DIAMETER_PT / 2 + 0.3,
            }
        ax.annotate(**annotation)

    ax.axvline(1, color="#BBBBBB", linestyle=(0, (3, 3)), lw=0.7, zorder=0.4)
    ax.axhline(0, color="#BBBBBB", linestyle=(0, (3, 3)), lw=0.7, zorder=0.4)
    ax.set_xlim(0, 21.5)
    ax.set_ylim(-1.75, 0.90)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_yticks([-1.5, -1.0, -0.5, 0, 0.5])
    ax.set_xlabel("Signal gain (fold vs. 2 probes)", fontsize=10, labelpad=5)
    ax.set_ylabel("FPR change\n[log2(fold vs. 2 probes)]", fontsize=10, labelpad=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.8, width=0.7, pad=2, labelsize=8)

    _add_e_legends(ax)
    paths = _save_panel(fig, panel_dir, "panel_E")
    return {
        "paths": paths,
        "table": export_dir / "panel_E_points.csv",
    }


def generate_all(workbook: Path, output_dir: Path) -> dict[str, Any]:
    """Generate B/C/E quantitative panels from the supplied XLSX workbook."""

    workbook = Path(workbook).resolve()
    output_dir = Path(output_dir).resolve()
    _set_style()
    counts, line, quadrant, _ = _validate_workbook(workbook)
    panel_dir = output_dir / "panels"
    export_dir = output_dir / "exports" / "quantitative"
    panel_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)
    return {
        "B": _plot_panel_b(counts, panel_dir, export_dir),
        "C": _plot_panel_c(line, panel_dir, export_dir),
        "E": _plot_panel_e(quadrant, panel_dir, export_dir),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workbook",
        type=Path,
        default=Path(__file__).resolve().with_name("fig1_data.xlsx"),
        help="XLSX workbook; default is fig1_data.xlsx beside this script.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory receiving generated panels and summary tables.",
    )
    args = parser.parse_args()
    generate_all(args.workbook, args.output_dir)
