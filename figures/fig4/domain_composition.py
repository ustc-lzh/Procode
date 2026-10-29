#!/usr/bin/env python3
"""Final-domain composition statistics and plotting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_io import DOMAIN_ORDER, find_column


def recompute_composition(
    composition: pd.DataFrame | None,
    metadata: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame | None, dict]:
    """Compute P(cell type | final domain) and audit the packaged table."""

    checks: dict = {}
    required_metadata = {"retained_for_final_figure", "final_domain", "final_celltype_code"}
    missing_metadata = sorted(required_metadata.difference(metadata.columns))
    if missing_metadata:
        raise ValueError(
            "Domain composition requires metadata columns: " + ", ".join(missing_metadata) + "."
        )
    retained = metadata.loc[metadata["retained_for_final_figure"]].copy()
    if retained.empty:
        raise ValueError("Domain composition requires retained cells.")
    if retained["final_domain"].isna().any():
        raise ValueError("Retained metadata contains missing final_domain values.")
    rows: list[dict] = []
    labels = retained["final_celltype_code"].astype(str)
    domains = retained["final_domain"].astype(str)
    for domain in DOMAIN_ORDER:
        domain_rows = retained.loc[domains.eq(domain)]
        denominator = len(domain_rows)
        for label in config["celltype_order"]:
            n_cells = int(domain_rows["final_celltype_code"].astype(str).eq(label).sum())
            rows.append(
                {
                    "final_domain": domain,
                    "celltype": label,
                    "n_cells": n_cells,
                    "fraction_within_domain": n_cells / denominator if denominator else np.nan,
                    "celltype_display": config["display_map"].get(label, label),
                }
            )
    recomputed = pd.DataFrame(rows)
    if composition is None or composition.empty:
        checks["composition_source"] = "recomputed_from_canonical_metadata"
        return recomputed, checks

    source = composition.copy()
    if "final_domain" not in source:
        domain_column = find_column(source, "cast_region_merged", "domain")
        mapping = {
            "domain_4": "D1",
            "MES-enriched (domain_1 + domain_2)": "D2",
            "domain_0": "D3",
            "domain_3": "D4",
            "domain_5": "D5",
        }
        if domain_column:
            source["final_domain"] = source[domain_column].map(mapping)
    if "celltype" not in source:
        label_column = find_column(source, "broad_celltype", "final_celltype_code")
        if label_column:
            source = source.rename(columns={label_column: "celltype"})
    required_source = {
        "final_domain",
        "celltype",
        "n_cells",
        "fraction_within_domain",
    }
    missing_source = sorted(required_source.difference(source.columns))
    if missing_source:
        raise ValueError(
            "Precomputed domain composition requires columns: "
            + ", ".join(missing_source)
            + "."
        )
    source["final_domain"] = source["final_domain"].astype(str)
    source["celltype"] = source["celltype"].astype(str)
    merged = recomputed.merge(source, on=["final_domain", "celltype"], suffixes=("_recomputed", "_source"))
    if len(merged) != len(recomputed):
        raise ValueError("Precomputed domain composition does not cover all final-domain/type pairs.")
    n_diff = np.max(
        np.abs(
            pd.to_numeric(merged["n_cells_recomputed"], errors="coerce").to_numpy()
            - pd.to_numeric(merged["n_cells_source"], errors="coerce").to_numpy()
        )
    )
    f_diff = np.nanmax(
        np.abs(
            pd.to_numeric(merged["fraction_within_domain_recomputed"], errors="coerce").to_numpy()
            - pd.to_numeric(merged["fraction_within_domain_source"], errors="coerce").to_numpy()
        )
    )
    checks["composition_source"] = "recomputed_from_canonical_metadata"
    checks["composition_table_agrees"] = bool(n_diff == 0 and f_diff <= 1e-8)
    checks["composition_max_fraction_difference"] = float(f_diff)
    if not checks["composition_table_agrees"]:
        raise ValueError("Recomputed final-domain composition disagrees with the supplied source table.")
    return recomputed, checks


def plot_composition(ax, composition: pd.DataFrame | None, config: dict) -> None:
    """Draw the stacked final-domain composition panel."""

    if composition is None or composition.empty:
        ax.set_title("Final-domain composition")
        ax.text(0.5, 0.5, "composition table unavailable", transform=ax.transAxes, ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return
    table = composition.copy()
    required = {"final_domain", "celltype", "fraction_within_domain"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError("Final-domain composition requires columns: " + ", ".join(missing) + ".")
    table["final_domain"] = table["final_domain"].astype(str)
    table["celltype"] = table["celltype"].astype(str)
    raw_fraction = table["fraction_within_domain"]
    table["fraction_within_domain"] = pd.to_numeric(raw_fraction, errors="coerce")
    invalid_fraction = raw_fraction.notna() & (
        table["fraction_within_domain"].isna()
        | ~np.isfinite(table["fraction_within_domain"])
    )
    if invalid_fraction.any():
        raise ValueError("Final-domain composition fractions must be finite numbers.")
    finite_fraction = table["fraction_within_domain"].dropna()
    if ((finite_fraction < 0) | (finite_fraction > 1)).any():
        raise ValueError("Final-domain composition fractions must be between 0 and 1.")
    left = np.zeros(len(DOMAIN_ORDER), dtype=float)
    y = np.arange(len(DOMAIN_ORDER))
    for label in config["celltype_order"]:
        values = []
        for domain in DOMAIN_ORDER:
            row = table.loc[table["final_domain"].eq(domain) & table["celltype"].eq(label)]
            value = row.iloc[0]["fraction_within_domain"] if not row.empty else 0.0
            values.append(0.0 if pd.isna(value) else float(value))
        values = np.asarray(values)
        ax.barh(
            y,
            values,
            left=left,
            height=0.72,
            color=config["palette"].get(label, "#777777"),
            label=config["display_map"].get(label, label),
        )
        left += values
    ax.set_yticks(y, DOMAIN_ORDER)
    ax.set_xlim(0, 1)
    ax.set_xlabel("P(cell type | final domain)")
    ax.set_title("Cell composition within final domains")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.02), frameon=False, ncol=1)

