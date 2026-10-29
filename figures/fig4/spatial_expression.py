#!/usr/bin/env python3
"""Coordinate-based transcript-position marker panel for Fig. 4 C."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_io import find_column


def plot_spots(ax, spots: pd.DataFrame | None, config: dict) -> None:
    """Render all valid tissue spots for the seven selected spatial markers."""

    if spots is None:
        raise FileNotFoundError("Transcript-position data are required for the marker-position panel.")
    if spots.empty:
        raise ValueError("Transcript-position data are empty.")
    gene_column = find_column(spots, "gene", "target_gene")
    x_column = find_column(spots, "x_adjust", "x", "x_position")
    y_column = find_column(spots, "y_adjust", "y", "y_position")
    if not gene_column or not x_column or not y_column:
        missing = [
            name
            for name, column in (("gene", gene_column), ("x", x_column), ("y", y_column))
            if column is None
        ]
        raise ValueError(
            "Marker transcript positions require columns: " + ", ".join(missing) + "."
        )
    marker_to_label = {gene: label for label, gene in config["marker_map"].items()}
    selected = spots[gene_column].astype(str).isin(marker_to_label)
    plot = spots.loc[selected, [gene_column, x_column, y_column]].copy()
    if plot.empty:
        raise ValueError("Transcript-position data contain none of the configured spatial marker genes.")
    plot["display_x_um"] = pd.to_numeric(plot[y_column], errors="coerce") * config["pixel_size_um"]
    plot["display_y_um"] = pd.to_numeric(plot[x_column], errors="coerce") * config["pixel_size_um"]
    plot = plot.dropna(subset=["display_x_um", "display_y_um"])
    plot = plot.loc[np.isfinite(plot[["display_x_um", "display_y_um"]]).all(axis=1)]
    if plot.empty:
        raise ValueError("Configured spatial marker positions have no finite coordinates.")
    for gene, label in marker_to_label.items():
        sub = plot.loc[plot[gene_column].astype(str).eq(gene)]
        if len(sub):
            ax.scatter(
                sub["display_x_um"],
                sub["display_y_um"],
                s=0.35,
                color=config["palette"].get(label, "#777777"),
                linewidths=0,
                alpha=0.62,
                rasterized=True,
                label=f"{config['display_map'].get(label, label)}: {gene}",
            )
    ax.set_title("Selected marker transcript positions")
    ax.set_xlabel("Display x (µm)")
    ax.set_ylabel("Display y (µm)")
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")

