#!/usr/bin/env python3
"""Shared workbook, release-data and configuration access for Fig. 4.

The workbook contains the compact, publication-facing tables.  Large tables
remain in the Result4 release directory or ZIP and are addressed by their
release-relative paths.  This module owns that boundary so plotting modules do
not silently fall back to legacy CSV/JSON layouts.
"""

from __future__ import annotations

import io
import json
from collections import OrderedDict
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd


SAMPLE = "20250821"
DOMAIN_ORDER = ["D1", "D2", "D3", "D4", "D5"]
DEFAULT_CELLTYPE_ORDER = [
    "AC-like",
    "MES-Ast",
    "MES-Hyp",
    "OPC-like",
    "NPC-like",
    "Myeloid",
    "T-HSPA1A",
    "Endo",
    "Peri/VSMC",
    "Oligo",
]
DEFAULT_DISPLAY = {
    "T-HSPA1A": "T cells",
    "Endo": "Endothelial",
    "Peri/VSMC": "Pericytes/VSMCs",
    "Oligo": "Oligodendrocytes",
}
DEFAULT_CELLTYPE_PALETTE = {
    "AC-like": "#FED100",
    "MES-Ast": "#4169E1",
    "MES-Hyp": "#D7008C",
    "OPC-like": "#00A555",
    "NPC-like": "#16CB55",
    "Myeloid": "#17728C",
    "T-HSPA1A": "#FFD9CE",
    "Endo": "#F51F1A",
    "Peri/VSMC": "#FC912A",
    "Oligo": "#AE958C",
}
DEFAULT_DOMAIN_PALETTE = {
    "D1": "#D45087",
    "D2": "#E76F51",
    "D3": "#4C78A8",
    "D4": "#7A5195",
    "D5": "#2A9D8F",
}
DEFAULT_SPATIAL_MARKERS = OrderedDict(
    [
        ("AC-like", "HOPX"),
        ("MES-Ast", "CHI3L1"),
        ("MES-Hyp", "VEGFA"),
        ("OPC-like", "GPR17"),
        ("NPC-like", "DCX"),
        ("Endo", "VWF"),
        ("Peri/VSMC", "COL1A2"),
    ]
)
DEFAULT_DOTPLOT_GENES = [
    "AQP4",
    "CLU",
    "CHI3L1",
    "TGFB2",
    "VEGFA",
    "ANGPTL4",
    "GPR17",
    "PDGFRA",
    "SOX11",
    "ELAVL4",
    "CD14",
    "TREM2",
    "IL7R",
    "DDIT3",
    "VWF",
    "PECAM1",
    "PDGFRB",
    "COL1A1",
    "COL1A2",
    "COL5A1",
    "CLDN11",
    "PLLP",
]


class DataLocator:
    """Read release-relative files from a directory or the supplied ZIP."""

    def __init__(self, root: Path | None):
        self.root = Path(root).expanduser().resolve() if root else None
        self.archive: zipfile.ZipFile | None = None
        self.prefix = ""
        if self.root and self.root.is_file() and self.root.suffix.lower() == ".zip":
            self.archive = zipfile.ZipFile(self.root)
            names = [name for name in self.archive.namelist() if name and not name.endswith("/")]
            prefixes = {name.split("/", 1)[0] for name in names if "/" in name}
            if len(prefixes) == 1:
                self.prefix = next(iter(prefixes)) + "/"

    def close(self) -> None:
        if self.archive is not None:
            self.archive.close()

    def _archive_name(self, relative: str) -> str:
        return self.prefix + relative.replace("\\", "/").lstrip("/")

    def exists(self, relative: str) -> bool:
        if self.archive is not None:
            try:
                self.archive.getinfo(self._archive_name(relative))
                return True
            except KeyError:
                return False
        return bool(self.root and (self.root / relative).is_file())

    def read_bytes(self, relative: str) -> bytes:
        if self.archive is not None:
            try:
                return self.archive.read(self._archive_name(relative))
            except KeyError as exc:
                raise FileNotFoundError(relative) from exc
        if not self.root:
            raise FileNotFoundError(relative)
        return (self.root / relative).read_bytes()

    def read_csv(self, relative: str, **kwargs) -> pd.DataFrame:
        if self.archive is not None:
            return pd.read_csv(io.BytesIO(self.read_bytes(relative)), **kwargs)
        if not self.root:
            raise FileNotFoundError(relative)
        return pd.read_csv(self.root / relative, **kwargs)

    def read_json(self, relative: str) -> object:
        return json.loads(self.read_bytes(relative).decode("utf-8"))


def norm_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def read_workbook(path: Path) -> dict[str, pd.DataFrame]:
    if not path.is_file():
        raise FileNotFoundError(f"Workbook not found: {path}")
    sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    return {norm_name(name): table for name, table in sheets.items()}


def sheet(sheets: dict[str, pd.DataFrame], *aliases: str) -> pd.DataFrame | None:
    for alias in aliases:
        table = sheets.get(norm_name(alias))
        if table is not None:
            return table.copy()
    return None


def read_table(
    sheets: dict[str, pd.DataFrame],
    locator: DataLocator,
    aliases: tuple[str, ...],
    external_names: tuple[str, ...],
) -> tuple[pd.DataFrame | None, str | None]:
    table = sheet(sheets, *aliases)
    if table is not None:
        return table, f"workbook:{aliases[0]}"
    for relative in external_names:
        if locator.exists(relative):
            return locator.read_csv(relative), relative
    return None, None


def json_value(value: object) -> object:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (dict, list, bool, int, float)):
        return value
    text = str(value).strip()
    if not text:
        return ""
    try:
        return json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return text


def load_settings(sheets: dict[str, pd.DataFrame], locator: DataLocator) -> tuple[dict, str]:
    table = sheet(sheets, "figure_settings", "settings", "figure_settings_map")
    settings: dict = {}
    source = "defaults"
    if table is not None and not table.empty:
        columns = {norm_name(column): column for column in table.columns}
        key_column = columns.get("key") or columns.get("setting") or columns.get("name")
        value_column = columns.get("value") or columns.get("settingvalue")
        if key_column and value_column:
            for row in table[[key_column, value_column]].itertuples(index=False):
                settings[str(row[0])] = json_value(row[1])
        elif len(table) == 1:
            settings = {str(k): json_value(v) for k, v in table.iloc[0].to_dict().items()}
        source = "workbook:figure_settings"
    elif locator.exists("metadata/figure_settings.json"):
        loaded = locator.read_json("metadata/figure_settings.json")
        if isinstance(loaded, dict):
            settings = loaded
        source = "metadata/figure_settings.json"
    return settings, source


def find_column(df: pd.DataFrame, *aliases: str) -> str | None:
    normalized = {norm_name(column): column for column in df.columns}
    for alias in aliases:
        if norm_name(alias) in normalized:
            return normalized[norm_name(alias)]
    return None


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    values = series.astype(str).str.strip().str.lower()
    return values.isin({"1", "true", "yes", "y", "t"})


def normalize_ids(values: pd.Series | pd.Index) -> pd.Index:
    return pd.Index(values).astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def _rename_known_columns(metadata: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "cell_id": ("cell_id", "cell", "cellid", "CellID"),
        "final_celltype_code": (
            "final_celltype_code",
            "broad_celltype",
            "final_celltype",
            "celltype_code",
            "celltype",
        ),
        "final_celltype_display": (
            "final_celltype_display",
            "celltype_display",
            "display_label",
        ),
        "final_domain": ("final_domain", "domain_final", "domain"),
        "retained_for_final_figure": (
            "retained_for_final_figure",
            "retained",
            "keep_for_final_figure",
        ),
        "in_projected_umap": ("in_projected_umap", "projected", "in_umap"),
    }
    rename: dict[str, str] = {}
    for canonical, names in aliases.items():
        found = find_column(metadata, *names)
        if found and found != canonical:
            rename[found] = canonical
    return metadata.rename(columns=rename)


def load_metadata(
    sheets: dict[str, pd.DataFrame], locator: DataLocator, settings: dict
) -> tuple[pd.DataFrame, str, dict]:
    metadata, source = read_table(
        sheets,
        locator,
        ("cell_metadata", "canonical_metadata", "cells", "metadata"),
        ("cells/cell_metadata.csv",),
    )
    if metadata is None:
        raise FileNotFoundError(
            "Canonical cell metadata is required as workbook sheet cell_metadata "
            "or data-root cells/cell_metadata.csv."
        )
    metadata = _rename_known_columns(metadata)
    if "cell_id" not in metadata:
        raise ValueError("Canonical metadata must contain cell_id.")
    metadata["cell_id"] = normalize_ids(metadata["cell_id"])
    if metadata["cell_id"].duplicated().any():
        raise ValueError("Canonical metadata contains duplicate cell_id values.")

    if "final_celltype_code" not in metadata:
        raise ValueError("Canonical metadata must contain final_celltype_code.")
    metadata["final_celltype_code"] = metadata["final_celltype_code"].replace(
        {"nan": np.nan, "None": np.nan, "": np.nan}
    )
    if "final_domain" not in metadata:
        merged_column = find_column(metadata, "cast_region_merged")
        domain_map = settings.get("final_domain_map", {})
        if merged_column and isinstance(domain_map, dict):
            metadata["final_domain"] = metadata[merged_column].map(domain_map)
        else:
            raise ValueError("Canonical metadata needs final_domain and final_domain_map.")
    if "retained_for_final_figure" not in metadata:
        raise ValueError("Canonical metadata must contain retained_for_final_figure.")
    metadata["retained_for_final_figure"] = as_bool(metadata["retained_for_final_figure"])
    if "in_projected_umap" not in metadata:
        required = {"UMAP_1", "UMAP_2"}
        if not required.issubset(metadata.columns):
            raise ValueError("Canonical metadata needs in_projected_umap or UMAP_1/UMAP_2.")
        metadata["in_projected_umap"] = metadata[["UMAP_1", "UMAP_2"]].notna().all(axis=1)
    else:
        metadata["in_projected_umap"] = as_bool(metadata["in_projected_umap"])
    if "final_celltype_display" not in metadata:
        display_map = settings.get("celltype_display_name_map", DEFAULT_DISPLAY)
        metadata["final_celltype_display"] = metadata["final_celltype_code"].replace(
            display_map if isinstance(display_map, dict) else DEFAULT_DISPLAY
        )

    checks: dict = {
        "metadata_rows": int(len(metadata)),
        "metadata_columns": int(len(metadata.columns)),
        "metadata_unique_ids": True,
        "retained_cells": int(metadata["retained_for_final_figure"].sum()),
        "projected_cells": int(metadata["in_projected_umap"].sum()),
    }
    populations = settings.get("populations", {})
    if isinstance(populations, dict):
        expected = {
            "raw_count_and_qc_cells": len(metadata),
            "projected_query_cells": int(metadata["in_projected_umap"].sum()),
            "retained_cells": int(metadata["retained_for_final_figure"].sum()),
        }
        for key, observed in expected.items():
            if key in populations:
                checks[f"{key}_matches_settings"] = int(populations[key]) == int(observed)

    retained = metadata.loc[metadata["retained_for_final_figure"]].copy()
    valid_domains = set(retained["final_domain"].dropna().astype(str))
    checks["retained_domains_are_D1_to_D5"] = valid_domains.issubset(set(DOMAIN_ORDER))
    if not checks["retained_domains_are_D1_to_D5"]:
        raise ValueError(f"Final retained metadata has non-final domains: {sorted(valid_domains)}")

    score_column = find_column(
        metadata,
        "legacy_prediction_score",
        "prediction.score.max",
        "prediction_score",
        "transfer_score",
    )
    if score_column:
        scores = pd.to_numeric(retained[score_column], errors="coerce")
        checks["retained_legacy_score_min"] = float(scores.min()) if len(scores) else None
        checks["retained_legacy_score_ge_0p5"] = bool(scores.ge(0.5).all())
        if not checks["retained_legacy_score_ge_0p5"]:
            raise ValueError("Retained cells violate the release legacy score threshold of 0.5.")
    legacy_label_column = find_column(
        metadata,
        "legacy_predicted_celltype",
        "predicted.id",
        "transferred_celltype",
    )
    if legacy_label_column:
        counts = retained[legacy_label_column].value_counts(dropna=False)
        checks["retained_legacy_label_min_count"] = int(counts.min()) if len(counts) else 0
        checks["retained_legacy_label_counts_ge_50"] = bool(counts.min() >= 50) if len(counts) else False

    return metadata, source or "unknown", checks


def load_reference(
    sheets: dict[str, pd.DataFrame], locator: DataLocator
) -> tuple[pd.DataFrame | None, str | None]:
    table, source = read_table(
        sheets,
        locator,
        ("reference_umap", "reference_panel_umap", "reference"),
        ("reference/reference_panel_umap.csv",),
    )
    if table is None:
        return None, None
    if "celltype" not in table:
        label_column = find_column(table, "cell_type", "annotation", "label")
        if label_column:
            table = table.rename(columns={label_column: "celltype"})
    for required in ("UMAP_1", "UMAP_2"):
        if required not in table:
            raise ValueError(f"Reference UMAP table lacks {required}.")
    return table, source


def load_counts(
    sheets: dict[str, pd.DataFrame], locator: DataLocator
) -> tuple[pd.DataFrame | None, str | None]:
    table = sheet(sheets, "counts", "matrix", "raw_counts")
    if table is not None:
        return table, "workbook:counts"
    for relative in ("counts/matrix.raw.csv", "matrix.raw.csv"):
        if locator.exists(relative):
            return locator.read_csv(relative), relative
    h5ad_name = next(
        (relative for relative in ("counts/matrix.h5ad", "counts/data.h5ad", "matrix.h5ad") if locator.exists(relative)),
        None,
    )
    if h5ad_name:
        try:
            import anndata  # type: ignore

            if locator.archive is not None:
                raise RuntimeError("Reading an H5AD entry from a ZIP is not streamed by this script.")
            obj = anndata.read_h5ad(locator.root / h5ad_name)
            dense = obj.X.toarray() if hasattr(obj.X, "toarray") else np.asarray(obj.X)
            return pd.DataFrame(dense, index=obj.obs_names, columns=obj.var_names).reset_index(
                names="cell"
            ), h5ad_name
        except (ImportError, RuntimeError, OSError, ValueError) as exc:
            raise RuntimeError(f"H5AD counts were supplied but could not be read: {exc}") from exc
    return None, None


def load_spots(
    sheets: dict[str, pd.DataFrame], locator: DataLocator
) -> tuple[pd.DataFrame | None, str | None]:
    return read_table(
        sheets,
        locator,
        ("transcript_positions", "gene_positions", "spots", "gene_position_filtered"),
        ("spatial/gene_position_filtered.csv", "gene_position_filtered.csv"),
    )


def load_roi(
    sheets: dict[str, pd.DataFrame], locator: DataLocator
) -> tuple[pd.DataFrame | None, str | None]:
    return read_table(
        sheets,
        locator,
        ("roi_centers", "rois", "roi"),
        ("spatial/roi_centers.csv", "roi_centers.csv"),
    )


def load_marker_table(
    sheets: dict[str, pd.DataFrame], locator: DataLocator
) -> tuple[pd.DataFrame | None, str | None]:
    return read_table(
        sheets,
        locator,
        ("marker_dotplot", "dotplot", "marker_stats"),
        ("source_data/marker_dotplot.csv",),
    )


def load_composition_table(
    sheets: dict[str, pd.DataFrame], locator: DataLocator
) -> tuple[pd.DataFrame | None, str | None]:
    return read_table(
        sheets,
        locator,
        ("domain_composition", "composition", "domain_stats"),
        ("source_data/domain_composition.csv",),
    )


def prepare_settings(settings: dict, marker_table: pd.DataFrame | None) -> dict:
    celltype_order = list(settings.get("celltype_order", DEFAULT_CELLTYPE_ORDER) or DEFAULT_CELLTYPE_ORDER)
    display_map = dict(settings.get("celltype_display_name_map", DEFAULT_DISPLAY) or DEFAULT_DISPLAY)
    palette = dict(settings.get("celltype_palette", DEFAULT_CELLTYPE_PALETTE) or DEFAULT_CELLTYPE_PALETTE)
    domain_palette = dict(settings.get("domain_palette", DEFAULT_DOMAIN_PALETTE) or DEFAULT_DOMAIN_PALETTE)
    marker_map = settings.get("spatial_C_markers", DEFAULT_SPATIAL_MARKERS)
    if not isinstance(marker_map, dict):
        marker_map = dict(DEFAULT_SPATIAL_MARKERS)
    marker_map = OrderedDict((label, str(marker_map[label])) for label in celltype_order if label in marker_map)
    genes = list(settings.get("dotplot_gene_order", DEFAULT_DOTPLOT_GENES) or DEFAULT_DOTPLOT_GENES)
    if marker_table is not None and "gene" in marker_table:
        source_genes = [str(x) for x in marker_table["gene"].dropna().drop_duplicates()]
        genes = [gene for gene in genes if gene in set(source_genes)] or source_genes
    return {
        "celltype_order": celltype_order,
        "display_map": display_map,
        "palette": palette,
        "domain_palette": domain_palette,
        "marker_map": marker_map,
        "dotplot_genes": genes,
        "pixel_size_um": float(settings.get("pixel_size_um", 0.1625) or 0.1625),
        "roi_side_um": float(settings.get("roi_side_um", 350.0) or 350.0),
    }


def normalize_roi_domains(roi: pd.DataFrame | None) -> pd.DataFrame | None:
    if roi is None:
        return None
    roi = roi.copy()
    if "final_domain" not in roi:
        region_column = find_column(roi, "region", "cast_region_merged", "domain")
        mapping = {
            "domain_4": "D1",
            "MES-enriched (domain_1 + domain_2)": "D2",
            "domain_0": "D3",
            "domain_3": "D4",
            "domain_5": "D5",
        }
        if region_column:
            roi["final_domain"] = roi[region_column].map(mapping)
    return roi

