#!/usr/bin/env python3
"""Independent Supplementary Fig. 6 QC, reference and composition panels."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_io import DOMAIN_ORDER
from domain_composition import plot_composition
from marker_expression import plot_dotplot


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix, kwargs in (("png", {"dpi": dpi}), ("pdf", {}), ("svg", {})):
        target = output_dir / f"{stem}.{suffix}"
        fig.savefig(target, bbox_inches="tight", pad_inches=0.03, **kwargs)
        paths.append(target.name)
    plt.close(fig)
    return paths


def _placeholder(ax: plt.Axes, title: str, detail: str) -> None:
    ax.set_title(title)
    ax.text(0.5, 0.5, detail, transform=ax.transAxes, ha="center", va="center", color="#666666")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_qc_hist(
    ax: plt.Axes,
    values: pd.Series,
    raw_mean: float,
    title: str,
    xlabel: str,
) -> None:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        _placeholder(ax, title, "QC values unavailable")
        return
    ax.hist(values, bins=36, color="#4F93C4", edgecolor="white", linewidth=0.25)
    ax.axvline(math.log2(raw_mean + 1), color="#D73027", linewidth=1.3)
    ax.text(
        0.98,
        0.94,
        f"Average = {raw_mean:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
        color="#B30000",
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cells")


def plot_reference(ax: plt.Axes, reference: pd.DataFrame | None) -> None:
    if reference is None:
        _placeholder(ax, "GBM scRNA-seq reference", "reference UMAP supplied separately")
        return
    ax.scatter(
        reference["UMAP_1"],
        reference["UMAP_2"],
        s=0.22,
        color="#BDBDBD",
        alpha=0.35,
        linewidths=0,
        rasterized=True,
    )
    ax.set_title("GBM scRNA-seq reference")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_aspect("equal", adjustable="box")


def plot_shared_umap(
    ax: plt.Axes,
    metadata: pd.DataFrame,
    reference: pd.DataFrame | None,
) -> None:
    if reference is not None:
        ax.scatter(
            reference["UMAP_1"],
            reference["UMAP_2"],
            s=0.20,
            color="#D1D1D1",
            alpha=0.25,
            linewidths=0,
            rasterized=True,
            label="GBM scRNA-seq reference",
        )
    query = metadata.loc[metadata["in_projected_umap"]].dropna(subset=["UMAP_1", "UMAP_2"])
    excluded = query.loc[~query["retained_for_final_figure"]]
    retained = query.loc[query["retained_for_final_figure"]]
    if len(excluded):
        ax.scatter(
            excluded["UMAP_1"],
            excluded["UMAP_2"],
            s=0.28,
            color="#4C78A8",
            alpha=0.56,
            linewidths=0,
            rasterized=True,
            label="Excluded",
        )
    if len(retained):
        ax.scatter(
            retained["UMAP_1"],
            retained["UMAP_2"],
            s=0.30,
            color="#E64B35",
            alpha=0.62,
            linewidths=0,
            rasterized=True,
            label="Retained",
        )
    fraction = len(retained) / max(len(query), 1)
    ax.set_title(f"Projected query in shared UMAP ({len(retained):,}/{len(query):,}, {fraction:.1%})")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_aspect("equal", adjustable="box")


def render_sfig6_outputs(
    metadata: pd.DataFrame,
    reference: pd.DataFrame | None,
    marker_table: pd.DataFrame | None,
    composition: pd.DataFrame | None,
    config: dict,
    output_dir: Path,
    dpi: int,
) -> list[str]:
    """Write independent Supplementary Fig. 6 artwork, never a page layout."""

    outputs: list[str] = []
    genes = metadata.get("genes_per_cell", pd.Series(dtype=float))
    transcripts = metadata.get("transcripts_per_cell", pd.Series(dtype=float))
    qc_specs = (
        (
            "SFig6_qc_genes_per_cell",
            metadata.get("log2_genes_per_cell_plus1", np.log2(pd.to_numeric(genes, errors="coerce") + 1)),
            float(pd.to_numeric(genes, errors="coerce").mean()),
            "Genes per cell",
            "log2(genes per cell + 1)",
        ),
        (
            "SFig6_qc_transcripts_per_cell",
            metadata.get("log2_transcripts_per_cell_plus1", np.log2(pd.to_numeric(transcripts, errors="coerce") + 1)),
            float(pd.to_numeric(transcripts, errors="coerce").mean()),
            "Transcripts per cell",
            "log2(transcripts per cell + 1)",
        ),
    )
    for stem, values, mean, title, xlabel in qc_specs:
        figure, ax = plt.subplots(figsize=(4.6, 3.6))
        plot_qc_hist(ax, values, mean, title, xlabel)
        outputs.extend(save_figure(figure, output_dir, stem, dpi))
    figure, ax = plt.subplots(figsize=(5.0, 4.4))
    plot_shared_umap(ax, metadata, reference)
    outputs.extend(save_figure(figure, output_dir, "SFig6_shared_umap", dpi))
    figure, ax = plt.subplots(figsize=(8.0, 4.8))
    plot_dotplot(ax, marker_table, config)
    outputs.extend(save_figure(figure, output_dir, "SFig6_marker_dotplot", dpi))
    figure, ax = plt.subplots(figsize=(5.4, 3.8))
    plot_composition(ax, composition, config)
    outputs.extend(save_figure(figure, output_dir, "SFig6_domain_composition", dpi))
    return outputs

