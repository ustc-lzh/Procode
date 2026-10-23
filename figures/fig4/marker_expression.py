#!/usr/bin/env python3
"""Retained-cell marker statistics and independent dot-plot rendering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_io import DOMAIN_ORDER, find_column
from gene_expression import log_normalize_34, retained_counts


def recompute_markers(
    marker_table: pd.DataFrame | None,
    counts: pd.DataFrame | None,
    metadata: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame | None, dict]:
    """Recompute and audit the workbook marker table when raw counts are present."""

    checks: dict = {}
    if marker_table is None and counts is None:
        checks["marker_stats_source"] = "unavailable"
        return None, checks
    if marker_table is None:
        raise ValueError("Counts are available but marker_dotplot source table is missing.")
    marker_table = marker_table.copy()
    if counts is None:
        checks["marker_stats_source"] = "precomputed:marker_dotplot"
        return marker_table, checks

    genes = list(config["dotplot_genes"])
    selected_counts, retained = retained_counts(counts, metadata, genes)
    normalized = log_normalize_34(selected_counts, retained["transcripts_per_cell"])
    labels = retained["final_celltype_code"].astype(str)
    rows: list[dict] = []
    source_owner: dict[str, str] = {}
    if "marker_owner" in marker_table:
        source_owner = dict(
            zip(marker_table["gene"].astype(str), marker_table["marker_owner"].astype(str))
        )
    for label in config["celltype_order"]:
        selected = labels.eq(label).to_numpy()
        if not selected.any():
            continue
        for gene in genes:
            values = selected_counts.loc[selected, gene].to_numpy(dtype=float)
            mean_values = normalized.loc[selected, gene].to_numpy(dtype=float)
            rows.append(
                {
                    "celltype": label,
                    "gene": gene,
                    "marker_owner": source_owner.get(gene, ""),
                    "n_cells": int(selected.sum()),
                    "fraction_positive": float((values > 0).mean()),
                    "mean_log_normalized": float(mean_values.mean()),
                }
            )
    recomputed = pd.DataFrame(rows)
    recomputed["scaled_mean_expression"] = 0.0
    for gene, positions in recomputed.groupby("gene", sort=False).groups.items():
        values = recomputed.loc[positions, "mean_log_normalized"]
        span = float(values.max() - values.min())
        if span > 0:
            recomputed.loc[positions, "scaled_mean_expression"] = (values - values.min()) / span
    if "celltype_display" in marker_table:
        displays = marker_table[["celltype", "celltype_display"]].drop_duplicates("celltype")
        recomputed = recomputed.merge(displays, on="celltype", how="left")
    else:
        recomputed["celltype_display"] = recomputed["celltype"].replace(config["display_map"])

    compare_columns = [
        "n_cells",
        "fraction_positive",
        "mean_log_normalized",
        "scaled_mean_expression",
    ]
    source = marker_table.copy()
    source["celltype"] = source["celltype"].astype(str)
    source["gene"] = source["gene"].astype(str)
    merged = recomputed.merge(source, on=["celltype", "gene"], suffixes=("_recomputed", "_source"))
    if len(merged) != len(recomputed):
        raise ValueError("Precomputed marker table does not cover all cell-type/gene pairs.")
    differences = []
    for column in compare_columns:
        left = pd.to_numeric(merged[f"{column}_recomputed"], errors="coerce")
        right = pd.to_numeric(merged[f"{column}_source"], errors="coerce")
        differences.append(float(np.nanmax(np.abs(left.to_numpy() - right.to_numpy()))))
    checks["marker_stats_source"] = "recomputed_from_counts"
    checks["marker_table_agrees"] = bool(max(differences, default=0.0) <= 1e-4)
    checks["marker_table_max_abs_difference"] = max(differences, default=0.0)
    if not checks["marker_table_agrees"]:
        raise ValueError(
            "Recomputed marker statistics disagree with the supplied source: "
            f"{checks['marker_table_max_abs_difference']}"
        )
    return recomputed, checks


def plot_dotplot(ax: plt.Axes, marker_table: pd.DataFrame | None, config: dict) -> None:
    """Plot the independent marker expression panel."""

    if marker_table is None:
        ax.set_title("Marker expression")
        ax.text(0.5, 0.5, "marker table unavailable", transform=ax.transAxes, ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return
    table = marker_table.copy()
    for column in ("celltype", "gene"):
        if column not in table:
            ax.set_title("Marker expression")
            ax.text(0.5, 0.5, f"{column} column unavailable", transform=ax.transAxes, ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            return
    table["celltype"] = table["celltype"].astype(str)
    table["gene"] = table["gene"].astype(str)
    present_genes = set(table["gene"])
    genes = [gene for gene in config["dotplot_genes"] if gene in present_genes]
    present_labels = set(table["celltype"])
    labels = [label for label in config["celltype_order"] if label in present_labels]
    positions = []
    colors = []
    sizes = []
    for yi, label in enumerate(labels):
        for xi, gene in enumerate(genes):
            row = table.loc[table["celltype"].eq(label) & table["gene"].eq(gene)]
            if row.empty:
                continue
            item = row.iloc[0]
            fraction = float(item.get("fraction_positive", 0.0))
            scaled = float(item.get("scaled_mean_expression", 0.0))
            positions.append((xi, yi))
            colors.append(scaled)
            sizes.append(0.0 if fraction <= 0 else 3.0 + 95.0 * fraction**0.9)
    if not positions:
        ax.set_title("Marker expression")
        ax.text(0.5, 0.5, "no marker rows available", transform=ax.transAxes, ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return
    xy = np.asarray(positions)
    scatter = ax.scatter(
        xy[:, 0], xy[:, 1], c=colors, s=sizes, cmap="viridis", vmin=0, vmax=1,
        linewidths=0.18, edgecolors="#444444",
    )
    ax.set_xticks(range(len(genes)), genes, rotation=70, ha="right")
    ax.set_yticks(range(len(labels)), [config["display_map"].get(label, label) for label in labels])
    ax.set_xlim(-0.6, len(genes) - 0.4)
    ax.set_ylim(len(labels) - 0.4, -0.6)
    ax.set_title("Marker expression in retained cells")
    ax.set_xlabel("Gene")
    ax.set_ylabel("Cell type")
    colorbar = ax.figure.colorbar(scatter, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label("Scaled mean expression", fontsize=6)

