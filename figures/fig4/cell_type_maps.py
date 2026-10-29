#!/usr/bin/env python3
"""Spatial cell-type and projected-query UMAP panels."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def _require_columns(table: pd.DataFrame, required: set[str], panel: str) -> None:
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"{panel} requires columns: {', '.join(missing)}.")


def _finite_rows(table: pd.DataFrame, columns: tuple[str, str], panel: str) -> pd.DataFrame:
    numeric = table.assign(
        **{column: pd.to_numeric(table[column], errors="coerce") for column in columns}
    )
    numeric = numeric.dropna(subset=list(columns))
    numeric = numeric.loc[np.isfinite(numeric[list(columns)]).all(axis=1)]
    if numeric.empty:
        raise ValueError(f"{panel} has no finite coordinate pairs.")
    return numeric


def plot_spatial_celltypes(ax: plt.Axes, retained: pd.DataFrame, config: dict) -> None:
    _require_columns(
        retained,
        {"display_x_um", "display_y_um", "final_celltype_code"},
        "Spatial cell-type panel",
    )
    retained = _finite_rows(retained, ("display_x_um", "display_y_um"), "Spatial cell-type panel")
    for label in config["celltype_order"]:
        selected = retained["final_celltype_code"].astype(str).eq(label)
        sub = retained.loc[selected].dropna(subset=["display_x_um", "display_y_um"])
        if len(sub):
            ax.scatter(
                sub["display_x_um"],
                sub["display_y_um"],
                s=0.42,
                color=config["palette"].get(label, "#777777"),
                linewidths=0,
                alpha=0.72,
                rasterized=True,
                label=config["display_map"].get(label, label),
            )
    ax.set_title("Spatial cell types")
    ax.set_xlabel("Display x (µm)")
    ax.set_ylabel("Display y (µm)")
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")


def plot_query_umap(ax: plt.Axes, metadata: pd.DataFrame, config: dict) -> None:
    _require_columns(
        metadata,
        {
            "UMAP_1",
            "UMAP_2",
            "in_projected_umap",
            "retained_for_final_figure",
            "final_celltype_code",
        },
        "Projected query UMAP panel",
    )
    query = metadata.loc[metadata["in_projected_umap"]].copy()
    query = _finite_rows(query, ("UMAP_1", "UMAP_2"), "Projected query UMAP panel")
    for label in config["celltype_order"]:
        sub = query.loc[query["final_celltype_code"].astype(str).eq(label)]
        if len(sub):
            ax.scatter(
                sub["UMAP_1"],
                sub["UMAP_2"],
                s=0.32,
                color=config["palette"].get(label, "#777777"),
                linewidths=0,
                alpha=0.67,
                rasterized=True,
            )
    excluded = query.loc[~query["retained_for_final_figure"]]
    if len(excluded):
        ax.scatter(
            excluded["UMAP_1"],
            excluded["UMAP_2"],
            s=0.22,
            color="#BDBDBD",
            linewidths=0,
            alpha=0.42,
            rasterized=True,
        )
    ax.set_title("Projected query UMAP")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_aspect("equal", adjustable="box")

