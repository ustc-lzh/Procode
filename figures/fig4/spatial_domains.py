#!/usr/bin/env python3
"""Final D1-D5 spatial-domain and ROI panels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_io import DOMAIN_ORDER


def _require_columns(table: pd.DataFrame, required: set[str], panel: str) -> None:
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"{panel} requires columns: {', '.join(missing)}.")


def _finite_float(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite number.") from None
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite number.")
    return result


def _positive_float(value: object, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return result


def _finite_coordinates(table: pd.DataFrame, panel: str) -> pd.DataFrame:
    numeric = table.assign(
        **{
            column: pd.to_numeric(table[column], errors="coerce")
            for column in ("display_x_um", "display_y_um")
        }
    )
    numeric = numeric.dropna(subset=["display_x_um", "display_y_um"])
    numeric = numeric.loc[np.isfinite(numeric[["display_x_um", "display_y_um"]]).all(axis=1)]
    if numeric.empty:
        raise ValueError(f"{panel} has no finite coordinate pairs.")
    return numeric


def plot_domains(ax, retained: pd.DataFrame, config: dict) -> None:
    _require_columns(
        retained,
        {"display_x_um", "display_y_um", "final_domain"},
        "Final spatial-domain panel",
    )
    retained = _finite_coordinates(retained, "Final spatial-domain panel")
    if retained["final_domain"].isna().any():
        raise ValueError("Final spatial-domain panel has missing final_domain values.")
    observed_domains = set(retained["final_domain"].dropna().astype(str))
    unknown_domains = sorted(observed_domains.difference(DOMAIN_ORDER))
    if unknown_domains:
        raise ValueError(f"Final spatial-domain panel has unknown domains: {unknown_domains}.")
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
        raise FileNotFoundError(f"ROI table has no data for final domain {domain}.")
    required = {"display_x_um", "display_y_um", "final_celltype_code"}
    _require_columns(retained, required, f"ROI {domain} panel")
    retained = _finite_coordinates(retained, f"ROI {domain} panel")
    _require_columns(roi, {"final_domain", "center_x_um", "center_y_um"}, f"ROI {domain} panel")
    row = roi.loc[roi["final_domain"].astype(str).eq(domain)]
    if row.empty:
        raise FileNotFoundError(f"ROI table has no row for final domain {domain}.")
    item = row.iloc[0]
    center_x = _finite_float(item["center_x_um"], f"ROI {domain} center_x_um")
    center_y = _finite_float(item["center_y_um"], f"ROI {domain} center_y_um")
    width = _positive_float(item.get("roi_width_um", config["roi_side_um"]), f"ROI {domain} width")
    height = _positive_float(item.get("roi_height_um", config["roi_side_um"]), f"ROI {domain} height")
    xmin = _finite_float(item.get("xmin_um", center_x - width / 2), f"ROI {domain} xmin_um")
    xmax = _finite_float(item.get("xmax_um", center_x + width / 2), f"ROI {domain} xmax_um")
    ymin = _finite_float(item.get("ymin_um", center_y - height / 2), f"ROI {domain} ymin_um")
    ymax = _finite_float(item.get("ymax_um", center_y + height / 2), f"ROI {domain} ymax_um")
    if not xmax > xmin or not ymax > ymin:
        raise ValueError(f"ROI {domain} bounds must have positive width and height.")
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

