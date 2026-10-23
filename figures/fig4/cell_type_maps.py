#!/usr/bin/env python3
"""Standalone spatial cell-type and projected-query UMAP panels."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def placeholder(ax: plt.Axes, title: str, detail: str) -> None:
    ax.set_title(title)
    ax.text(0.5, 0.5, detail, transform=ax.transAxes, ha="center", va="center", color="#666666")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_spatial_celltypes(ax: plt.Axes, retained: pd.DataFrame, config: dict) -> None:
    if not {"display_x_um", "display_y_um"}.issubset(retained.columns):
        placeholder(ax, "Spatial cell types", "display coordinates unavailable")
        return
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
    required = {"UMAP_1", "UMAP_2"}
    if not required.issubset(metadata.columns):
        placeholder(ax, "Projected query UMAP", "UMAP coordinates unavailable")
        return
    query = metadata.loc[metadata["in_projected_umap"]].dropna(subset=["UMAP_1", "UMAP_2"])
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

