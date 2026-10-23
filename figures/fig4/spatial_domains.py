#!/usr/bin/env python3
"""Final D1-D5 spatial-domain and ROI panels."""

from __future__ import annotations

import pandas as pd

from data_io import DOMAIN_ORDER
from cell_type_maps import placeholder


def plot_domains(ax, retained: pd.DataFrame, config: dict) -> None:
    if not {"display_x_um", "display_y_um", "final_domain"}.issubset(retained.columns):
        placeholder(ax, "Final spatial domains", "final-domain coordinates unavailable")
        return
    for domain in DOMAIN_ORDER:
        selected = retained["final_domain"].astype(str).eq(domain)
        sub = retained.loc[selected].dropna(subset=["display_x_um", "display_y_um"])
        if len(sub):
            ax.scatter(
                sub["display_x_um"],
                sub["display_y_um"],
                s=0.45,
                color=config["domain_palette"].get(domain, "#777777"),
                linewidths=0,
                alpha=0.72,
                rasterized=True,
                label=domain,
            )
    ax.set_title("Final CAST domains")
    ax.set_xlabel("Display x (µm)")
    ax.set_ylabel("Display y (µm)")
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")


def plot_single_roi(
    ax,
    retained: pd.DataFrame,
    roi: pd.DataFrame | None,
    domain: str,
    config: dict,
) -> None:
    """Render one coordinate ROI without mask-based segmentation."""

    if roi is None or roi.empty:
        placeholder(ax, f"ROI {domain}", "ROI coordinates unavailable")
        return
    required = {"display_x_um", "display_y_um", "final_celltype_code"}
    if not required.issubset(retained.columns):
        placeholder(ax, f"ROI {domain}", "cell coordinates unavailable")
        return
    row = roi.loc[roi["final_domain"].astype(str).eq(domain)]
    if row.empty:
        placeholder(ax, f"ROI {domain}", "ROI unavailable")
        return
    item = row.iloc[0]
    try:
        center_x = float(item["center_x_um"])
        center_y = float(item["center_y_um"])
    except (KeyError, TypeError, ValueError):
        placeholder(ax, f"ROI {domain}", "ROI center unavailable")
        return
    width = float(item.get("roi_width_um", config["roi_side_um"]))
    height = float(item.get("roi_height_um", config["roi_side_um"]))
    xmin = float(item.get("xmin_um", center_x - width / 2))
    xmax = float(item.get("xmax_um", center_x + width / 2))
    ymin = float(item.get("ymin_um", center_y - height / 2))
    ymax = float(item.get("ymax_um", center_y + height / 2))
    inside = retained[
        retained["display_x_um"].between(xmin, xmax)
        & retained["display_y_um"].between(ymin, ymax)
    ]
    for label in config["celltype_order"]:
        sub = inside.loc[inside["final_celltype_code"].astype(str).eq(label)]
        if len(sub):
            ax.scatter(
                sub["display_x_um"],
                sub["display_y_um"],
                s=0.68,
                color=config["palette"].get(label, "#777777"),
                linewidths=0,
                alpha=0.72,
                rasterized=True,
                label=config["display_map"].get(label, label),
            )
    ax.set_title(f"ROI {domain}")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(config["domain_palette"].get(domain, "#777777"))
        spine.set_linewidth(1.0)

