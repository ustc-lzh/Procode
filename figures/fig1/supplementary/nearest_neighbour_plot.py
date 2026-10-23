"""Plot SFig2D from cached nearest-neighbour boxplot statistics.

The table contains the observed whisker endpoints from the original
1.5 x IQR rule. Distances and summary values are not recomputed here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

try:
    from data_io import SHEET_D, read_sheet
except ModuleNotFoundError:  # direct execution from the supplementary folder
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data_io import SHEET_D, read_sheet


PROBES = (2, 4, 8, 12)
LOW_COLOR = "#00AEEF"
HIGH_COLOR = "#ED1C24"
EDGE_COLOR = "#454545"

_REQUIRED_COLUMNS = {
    "group", "gene_class", "median_nn_dist_um", "mean_nn_dist_um",
    "q1_nn_dist_um", "q3_nn_dist_um", "whislo_um", "whishi_um",
}
_NUMERIC_COLUMNS = tuple(_REQUIRED_COLUMNS - {"group", "gene_class"})


def _load_statistics(stats_path: Path | str | pd.DataFrame) -> pd.DataFrame:
    """Load and validate the eight cached rows required by panel D."""

    if isinstance(stats_path, pd.DataFrame):
        path = "workbook sheet"
        frame = stats_path.copy()
    else:
        path = Path(stats_path)
        frame = pd.read_csv(path)
    missing = sorted(_REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    frame = frame.copy()
    frame["group"] = pd.to_numeric(frame["group"], errors="raise").astype(int)
    frame["gene_class"] = frame["gene_class"].astype(str).str.lower()
    expected = {
        (probe, gene_class)
        for probe in PROBES
        for gene_class in ("low_expression", "high_expression")
    }
    observed = set(zip(frame["group"], frame["gene_class"]))
    if observed != expected or len(frame) != len(expected):
        raise ValueError(
            "Panel D statistics must contain exactly one row for every "
            f"probe/class pair; observed {sorted(observed)}"
        )
    if frame.duplicated(["group", "gene_class"]).any():
        raise ValueError("Panel D statistics contain duplicate probe/class rows")

    for column in _NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Panel D statistics contain non-finite {column} values")

    for row in frame.itertuples(index=False):
        if not (
            row.whislo_um <= row.q1_nn_dist_um <= row.median_nn_dist_um
            <= row.q3_nn_dist_um <= row.whishi_um
        ):
            raise ValueError(
                "Expected whislo <= q1 <= median <= q3 <= whishi for "
                f"group={row.group}, gene_class={row.gene_class}"
            )
    if float(frame["whislo_um"].min()) < 0:
        raise ValueError("Panel D whisker lower endpoints must be non-negative")
    if float(frame["whishi_um"].max()) > 20.0:
        raise ValueError("Panel D data exceed the fixed 0-20 um axis")

    return frame.sort_values(["group", "gene_class"]).reset_index(drop=True)


def plot_panel_d(stats_path: Path | str | pd.DataFrame, output_path: Path | str) -> Path:
    """Draw cached boxplots; diamonds indicate the supplied mean distances."""

    statistics = _load_statistics(stats_path)
    output_path = Path(output_path)
    colors = {"low_expression": LOW_COLOR, "high_expression": HIGH_COLOR}
    labels = {
        "low_expression": "Low-expression genes",
        "high_expression": "High-expression genes",
    }
    offsets = {"low_expression": -0.18, "high_expression": 0.18}

    fig, axis = plt.subplots(figsize=(94 / 25.4, 84 / 25.4))
    for gene_class in ("low_expression", "high_expression"):
        rows = statistics.loc[statistics["gene_class"] == gene_class].set_index("group")
        positions = np.arange(len(PROBES)) + offsets[gene_class]
        boxes = [
            {
                "label": str(probe),
                "med": float(rows.loc[probe, "median_nn_dist_um"]),
                "q1": float(rows.loc[probe, "q1_nn_dist_um"]),
                "q3": float(rows.loc[probe, "q3_nn_dist_um"]),
                "whislo": float(rows.loc[probe, "whislo_um"]),
                "whishi": float(rows.loc[probe, "whishi_um"]),
                "fliers": [],
            }
            for probe in PROBES
        ]
        axis.bxp(
            boxes, positions=positions, widths=0.31, showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": colors[gene_class], "edgecolor": EDGE_COLOR},
            medianprops={"color": EDGE_COLOR},
            whiskerprops={"color": EDGE_COLOR},
            capprops={"color": EDGE_COLOR},
        )
        axis.scatter(
            positions, rows.loc[list(PROBES), "mean_nn_dist_um"],
            marker="D", s=12, facecolor="white", edgecolor=EDGE_COLOR, zorder=3,
        )

    axis.set(xlim=(-0.62, 3.62), ylim=(0, 20),
             xlabel="Probe number", ylabel="Nearest-neighbor distance (μm)")
    axis.set_xticks(range(len(PROBES)), PROBES)
    axis.set_yticks(np.arange(0, 20.1, 5))
    axis.spines[["top", "right"]].set_visible(False)
    handles = [
        Line2D([0], [0], color=colors[key], lw=5, label=labels[key])
        for key in ("low_expression", "high_expression")
    ]
    handles.append(Line2D([0], [0], marker="D", linestyle="none",
                          markerfacecolor="white", markeredgecolor=EDGE_COLOR,
                          markersize=4, label="Mean"))
    axis.legend(handles=handles, loc="upper right", frameon=False, fontsize=7)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor="white")
    if output_path.suffix.lower() == ".pdf":
        fig.savefig(output_path.with_suffix(".svg"), facecolor="white")
        fig.savefig(output_path.with_suffix(".png"), dpi=600, facecolor="white")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stats_path", type=Path, nargs="?")
    parser.add_argument("output_path", type=Path, nargs="?")
    parser.add_argument("--workbook", type=Path, default=None, help="Read sfig2_D_stats from an XLSX workbook.")
    parser.add_argument("--output", type=Path, default=None, help="Output PDF path when --workbook is used.")
    args = parser.parse_args()
    if args.workbook is not None:
        output = args.output or args.output_path
        if output is None:
            parser.error("--output is required with --workbook")
        plot_panel_d(read_sheet(args.workbook, SHEET_D), output)
    elif args.stats_path is None or args.output_path is None:
        parser.error("stats_path and output_path are required unless --workbook is used")
    else:
        plot_panel_d(args.stats_path, args.output_path)


if __name__ == "__main__":
    main()
