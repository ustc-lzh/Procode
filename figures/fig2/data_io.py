"""Input readers and shared definitions for the Figure 2 analyses.

The compact workbook is the versioned small input for this figure.  The two
large H5AD matrices stay external and are loaded only by stages that need
them.  Analysis-specific scoring and plotting deliberately live in the stage
modules rather than in this input layer.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd
from scipy import sparse


# The order is part of the Allen correlation definition and is retained in
# every regional source-data table.
REGIONS = [
    "CB", "HB", "HPF", "HY", "Isocortex", "CTXsp", "MB", "OLF", "PAL", "STR", "TH",
]
DISPLAY_REGIONS = [
    "CB", "CB_IP", "HB", "HPF", "HY", "Isocortex", "CTXsp", "MB", "OLF", "STR",
    "PAL", "TH", "TH_RT", "TH_SPFp", "fiber", "VL",
]
BITS = [f"{channel}_{round_id}" for round_id in range(1, 7) for channel in ("561", "647", "750")]


def _normalise_name(value: Any) -> str:
    return "".join(ch for ch in str(value).strip().lower() if ch.isalnum())


def _text(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _decode(values: Any) -> np.ndarray:
    return np.asarray([_text(value) for value in np.asarray(values)], dtype=object)


def _portable(value: Any) -> Any:
    """Convert NumPy values and non-finite floats before JSON serialisation."""

    if isinstance(value, np.generic):
        return _portable(value.item())
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, dict):
        return {str(key): _portable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_portable(value), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


class WorkbookInputs:
    """Resolve compact workbook sheets without loading unused sheets."""

    _ALIASES = {
        "background": (
            "background_peak_expression", "background_peaks", "reference_peak_expression",
            "reference_peaks", "full_reference_peaks", "background_peak",
        ),
        "target_expression": (
            "target_expression96", "target_expression_96", "target_expression",
            "target_expr", "target_expression_panel",
        ),
        "full_expression": (
            "expression_reference", "full_expression_reference", "brain_scRNA_celltype",
            "brain_scRNA_celltype_csv", "reference_expression", "expression",
        ),
        "abundance": (
            "cell_type_abundance", "celltype_abundance", "abundance",
            "brain_scRNA_abundance", "brain_scRNA_abundance_csv",
        ),
        "control": (
            "control_codebook", "codebook_control", "encode_refine_bad",
            "encode_bad", "control",
        ),
        "optimized": (
            "sa_optimized_codebook", "sa_optimized", "optimized_codebook",
            "codebook_sa_optimized", "encode_refine_good", "encode_good", "optimized",
        ),
        "allen": (
            "allen_reference", "allen_updated", "allen_syt2_arhgap12",
            "encode1_allen_syt2_arhgap12_updated", "allen",
        ),
    }

    def __init__(self, path: Path):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(
                f"Fig. 2 workbook not found: {self.path}. "
                "Place fig2_data.xlsx beside analysis.py or pass --workbook."
            )
        try:
            self.sheet_names = pd.ExcelFile(self.path).sheet_names
        except Exception as exc:  # pragma: no cover - parser-specific failure text
            raise ValueError(f"Cannot read Fig. 2 workbook {self.path}: {exc}") from exc
        self._by_normalised = {_normalise_name(name): name for name in self.sheet_names}

    def find(self, key: str, required: bool = True) -> str | None:
        aliases = self._ALIASES[key]
        for alias in aliases:
            found = self._by_normalised.get(_normalise_name(alias))
            if found is not None:
                return found
        # Permit descriptive suffixes while keeping exact aliases preferred.
        for alias in aliases:
            needle = _normalise_name(alias)
            for normalised, actual in self._by_normalised.items():
                if needle in normalised or normalised in needle:
                    return actual
        if required:
            names = ", ".join(self.sheet_names)
            raise ValueError(
                f"Workbook {self.path.name} is missing the '{key}' sheet. "
                f"Expected one of {aliases}; available sheets: {names}"
            )
        return None

    def read(self, key: str, required: bool = True) -> tuple[pd.DataFrame, str] | tuple[None, None]:
        sheet = self.find(key, required=required)
        if sheet is None:
            return None, None
        try:
            frame = pd.read_excel(self.path, sheet_name=sheet)
        except Exception as exc:  # pragma: no cover - parser-specific failure text
            raise ValueError(f"Cannot read workbook sheet '{sheet}': {exc}") from exc
        if frame.empty:
            raise ValueError(f"Workbook sheet '{sheet}' is empty")
        return frame, sheet


def _find_column(frame: pd.DataFrame, aliases: Iterable[str], *, required: bool = True) -> str | None:
    by_name = {_normalise_name(column): column for column in frame.columns}
    for alias in aliases:
        found = by_name.get(_normalise_name(alias))
        if found is not None:
            return found
    if required:
        raise ValueError(
            f"Cannot identify a required column in workbook sheet; columns are: "
            f"{list(map(str, frame.columns))}"
        )
    return None


def _id_column(frame: pd.DataFrame, aliases: Iterable[str]) -> str:
    column = _find_column(frame, aliases, required=False)
    if column is not None:
        return column
    return frame.columns[0]


def _clean_ids(values: Iterable[Any], label: str) -> pd.Index:
    result = pd.Index([_text(value).strip() for value in values], dtype="object")
    if result.isna().any() or (result == "").any() or (result == "nan").any():
        raise ValueError(f"Workbook {label} contains an empty identifier")
    return result


def _numeric_frame(frame: pd.DataFrame, columns: list[str], label: str) -> pd.DataFrame:
    try:
        numeric = frame[columns].apply(pd.to_numeric, errors="raise")
    except Exception as exc:
        raise ValueError(f"Workbook {label} has non-numeric values: {exc}") from exc
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"Workbook {label} must contain finite non-negative values")
    return numeric


# Escaped Unicode aliases preserve support for legacy non-English column headers.
def _load_expression_frame(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    id_column = _id_column(frame, ("gene", "gene_symbol", "gene_name", "\u57fa\u56e0", "\u57fa\u56e0\u540d\u79f0"))
    identifiers = _clean_ids(frame[id_column], label)
    excluded = {_normalise_name(id_column), "sourcerow", "row", "index", "sourceindex"}
    columns: list[str] = []
    for column in frame.columns:
        if _normalise_name(column) in excluded:
            continue
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.notna().all():
            columns.append(column)
    if len(columns) != 34:
        raise ValueError(
            f"Workbook {label} must contain 34 cell-type expression columns; found {len(columns)}"
        )
    numeric = _numeric_frame(frame, columns, label)
    numeric.index = identifiers
    numeric.index.name = "Gene"
    return numeric


def _load_codebook(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    id_column = _id_column(frame, ("gene", "gene_symbol", "gene_name", "\u57fa\u56e0", "\u57fa\u56e0\u540d\u79f0"))
    identifiers = _clean_ids(frame[id_column], label)
    columns = [column for column in frame.columns if re.fullmatch(r"(?:561|647|750)_\d+", str(column))]
    if len(columns) != 18:
        # Excel exports sometimes stringify numeric-looking bit names. Keep a
        # strict binary fallback only when there are exactly 18 numeric columns.
        excluded = {_normalise_name(id_column), "sourcerow", "row", "index"}
        candidates = []
        for column in frame.columns:
            if _normalise_name(column) in excluded:
                continue
            converted = pd.to_numeric(frame[column], errors="coerce")
            if converted.notna().all():
                candidates.append(column)
        if len(candidates) == 18:
            columns = candidates
    if len(columns) != 18:
        raise ValueError(f"Workbook {label} must contain exactly 18 binary bit columns")
    numeric = _numeric_frame(frame, columns, label)
    values = numeric.to_numpy(dtype=float)
    if not np.isin(values, (0, 1)).all():
        raise ValueError(f"Workbook {label} bit values must be 0 or 1")
    if not np.all(values.sum(axis=1) == 2):
        raise ValueError(f"Workbook {label} must assign exactly two bits to each gene")
    numeric = numeric.astype(int)
    numeric.index = identifiers
    numeric.index.name = "Gene"
    return numeric


def _load_abundance(frame: pd.DataFrame, expression_columns: pd.Index) -> pd.Series:
    id_column = _id_column(frame, ("celltype", "cell_type", "cell type", "\u7ec6\u80de\u7c7b\u578b"))
    value_column = _find_column(frame, ("class", "abundance", "count", "value", "\u6570\u91cf"), required=False)
    if value_column is None:
        candidates = [column for column in frame.columns if column != id_column]
        if len(candidates) != 1:
            raise ValueError("Workbook abundance sheet must contain one numeric abundance column")
        value_column = candidates[0]
    identifiers = _clean_ids(frame[id_column], "abundance")
    values = pd.to_numeric(frame[value_column], errors="raise")
    if not np.isfinite(values.to_numpy(dtype=float)).all() or (values < 0).any():
        raise ValueError("Workbook abundance values must be finite and non-negative")
    abundance = pd.Series(values.to_numpy(dtype=float), index=identifiers, name="Abundance")
    if not abundance.index.is_unique:
        raise ValueError("Workbook abundance identifiers are duplicated")
    missing = expression_columns.difference(abundance.index)
    extra = abundance.index.difference(expression_columns)
    if len(missing) or len(extra):
        raise ValueError(f"Abundance identifiers do not match expression columns; missing={list(missing)}, extra={list(extra)}")
    return abundance.reindex(expression_columns)


def _load_background_peaks(frame: pd.DataFrame) -> pd.DataFrame:
    gene_column = _id_column(frame, ("gene", "gene_symbol", "gene_name", "\u57fa\u56e0", "\u57fa\u56e0\u540d\u79f0"))
    peak_column = _find_column(
        frame, ("peakexpression", "peak_expression", "maxexpression", "max_expression", "peak", "\u8868\u8fbe\u5cf0\u503c"),
        required=False,
    )
    if peak_column is None:
        numeric_candidates = []
        for column in frame.columns:
            if column == gene_column or _normalise_name(column) in {"sourcerow", "row", "index"}:
                continue
            converted = pd.to_numeric(frame[column], errors="coerce")
            if converted.notna().all():
                numeric_candidates.append(column)
        if len(numeric_candidates) != 1:
            raise ValueError("Background peak sheet must contain one PeakExpression column")
        peak_column = numeric_candidates[0]
    peaks = pd.to_numeric(frame[peak_column], errors="raise")
    if not np.isfinite(peaks.to_numpy(dtype=float)).all() or (peaks < 0).any():
        raise ValueError("Background PeakExpression values must be finite and non-negative")
    source_column = _find_column(frame, ("sourcerow", "source_row", "row"), required=False)
    result = pd.DataFrame({
        "SourceRow": frame[source_column] if source_column is not None else np.arange(len(frame), dtype=int),
        "Gene": _clean_ids(frame[gene_column], "background peaks"),
        "PeakExpression": peaks.to_numpy(dtype=float),
    })
    celltype_column = _find_column(frame, ("peakcelltype", "peak_cell_type", "maxcelltype"), required=False)
    if celltype_column is not None:
        result["PeakCellType"] = frame[celltype_column].map(_text)
    return result


def load_workbook_inputs(workbook: Path) -> dict[str, Any]:
    """Load all compact inputs needed by the crowding and regional stages."""

    book = WorkbookInputs(workbook)
    control_frame, control_sheet = book.read("control")
    optimized_frame, optimized_sheet = book.read("optimized")
    control = _load_codebook(control_frame, control_sheet)
    optimized = _load_codebook(optimized_frame, optimized_sheet)
    if len(control) != 96 or len(optimized) != 96:
        raise ValueError(f"Both codebooks must contain 96 target genes; got {len(control)} and {len(optimized)}")
    if set(control.index) != set(optimized.index):
        raise ValueError("Control and SA-optimized codebooks do not contain the same target genes")
    if list(control.columns) != list(optimized.columns):
        raise ValueError("Control and SA-optimized codebooks do not use the same bit columns")

    background_frame, background_sheet = book.read("background", required=False)
    target_frame, target_sheet = book.read("target_expression", required=False)
    full_frame, full_sheet = book.read("full_expression", required=False)
    if background_frame is None and full_frame is None:
        raise ValueError(
            "Workbook must provide background_peak_expression (all 30,934 reference rows) "
            "or a complete expression_reference sheet"
        )
    if target_frame is None and full_frame is None:
        raise ValueError("Workbook must provide target_expression96 or a complete expression_reference sheet")

    if background_frame is not None:
        background = _load_background_peaks(background_frame)
    else:
        full_expression_for_background = _load_expression_frame(full_frame, full_sheet)
        background = pd.DataFrame({
            "SourceRow": np.arange(len(full_expression_for_background), dtype=int),
            "Gene": full_expression_for_background.index.astype(str),
            "PeakExpression": full_expression_for_background.max(axis=1).to_numpy(dtype=float),
            "PeakCellType": full_expression_for_background.idxmax(axis=1).to_numpy(dtype=str),
        })
    if len(background) != 30934:
        raise ValueError(
            f"Background reference must retain all 30,934 original rows for P95; found {len(background)}"
        )
    if background["Gene"].nunique() != 30914:
        raise ValueError(
            "Background reference should contain 30,934 rows and 30,914 distinct gene symbols; "
            f"found {background['Gene'].nunique()} distinct symbols"
        )

    if target_frame is not None:
        target_expression = _load_expression_frame(target_frame, target_sheet)
    else:
        full_expression = _load_expression_frame(full_frame, full_sheet)
        target_expression = full_expression.loc[list(control.index)].copy()
    if len(target_expression) != 96 or not target_expression.index.is_unique:
        raise ValueError("Target expression sheet must contain exactly 96 unique target genes")
    if set(target_expression.index) != set(control.index):
        missing = sorted(set(control.index) - set(target_expression.index))
        extra = sorted(set(target_expression.index) - set(control.index))
        raise ValueError(f"Target expression genes differ from codebooks; missing={missing}, extra={extra}")
    target_expression = target_expression.reindex(control.index)
    abundance_frame, abundance_sheet = book.read("abundance")
    abundance = _load_abundance(abundance_frame, target_expression.columns)

    allen_frame, allen_sheet = book.read("allen")
    return {
        "workbook": str(Path(workbook).resolve()),
        "sheets": {
            "background": background_sheet or full_sheet,
            "target_expression": target_sheet or full_sheet,
            "abundance": abundance_sheet,
            "control": control_sheet,
            "optimized": optimized_sheet,
            "allen": allen_sheet,
        },
        "background": background,
        "target_expression": target_expression,
        "abundance": abundance,
        "books": {"Control": control, "SA-optimized": optimized},
        "allen_frame": allen_frame,
    }


def _read_categorical(group: h5py.Group) -> np.ndarray:
    if "categories" in group and "codes" in group:
        categories = _decode(group["categories"][:])
        codes = np.asarray(group["codes"][:], dtype=int)
        result = np.full(codes.shape, "__NA__", dtype=object)
        valid = codes >= 0
        result[valid] = categories[codes[valid]]
        return result
    return _decode(group[:])


def _h5_index_key(group: h5py.Group, fallback: str) -> str:
    return _text(group.attrs.get("_index", fallback))


def _read_h5_matrix(x_group: Any, shape: tuple[int, int]) -> sparse.csr_matrix:
    if isinstance(x_group, h5py.Dataset):
        return sparse.csr_matrix(np.asarray(x_group[:], dtype=float))
    encoding = _text(x_group.attrs.get("encoding-type", ""))
    if encoding and "csr" not in encoding.lower() and not {"data", "indices", "indptr"}.issubset(x_group.keys()):
        raise ValueError(f"Unsupported H5AD X encoding: {encoding}")
    return sparse.csr_matrix(
        (
            np.asarray(x_group["data"][:], dtype=float),
            np.asarray(x_group["indices"][:], dtype=np.int32),
            np.asarray(x_group["indptr"][:], dtype=np.int64),
        ),
        shape=shape,
    )


def _count_positive_per_row(matrix: sparse.csr_matrix) -> np.ndarray:
    cleaned = matrix.copy()
    cleaned.eliminate_zeros()
    return np.asarray(cleaned.getnnz(axis=1)).ravel().astype(np.int64)


def _replace_labels(labels: np.ndarray, mapping: dict[str, str]) -> np.ndarray:
    return np.asarray([mapping.get(str(value), str(value)) for value in labels], dtype=object)


def locate_h5ad_files(data_root: Path | None) -> tuple[Path, Path]:
    if data_root is None:
        raise FileNotFoundError(
            "Detection and regional stages require --data-root containing "
            "adata_bad_final_v2_260911.h5ad and adata_good_final_v2_260911.h5ad."
        )
    root = Path(data_root)
    candidates = [root, root / "inputs"]
    names = {
        "bad": "adata_bad_final_v2_260911.h5ad",
        "good": "adata_good_final_v2_260911.h5ad",
    }
    found: dict[str, Path] = {}
    for condition, name in names.items():
        for candidate_root in candidates:
            path = candidate_root / name
            if path.is_file():
                found[condition] = path
                break
    missing = [name for condition, name in names.items() if condition not in found]
    if missing:
        raise FileNotFoundError(
            f"Missing external H5AD file(s) under {root}: {', '.join(missing)}. "
            "Pass --data-root to the directory containing the two final H5AD files."
        )
    return found["bad"], found["good"]


def load_h5ad_dataset(path: Path, is_bad: bool) -> dict[str, Any]:
    """Read the required raw-count H5AD fields and apply the fixed QC rule."""

    if not path.is_file():
        raise FileNotFoundError(f"External H5AD file not found: {path}")
    try:
        with h5py.File(path, "r", locking=False) as handle:
            obs_group = handle["obs"]
            var_group = handle["var"]
            obs_key = _h5_index_key(obs_group, "_index")
            var_key = _h5_index_key(var_group, "_index")
            if obs_key not in obs_group:
                raise ValueError(f"{path.name} does not contain obs/{obs_key} cell identifiers")
            if var_key not in var_group:
                raise ValueError(f"{path.name} does not contain var/{var_key} gene identifiers")
            cell_ids_all = _decode(obs_group[obs_key][:])
            genes_all = _decode(var_group[var_key][:])
            regions_all = _read_categorical(obs_group["brain_region"])
            pixel_area_all = np.asarray(obs_group["pixel_area"][:], dtype=float)
            matrix_all = _read_h5_matrix(handle["X"], (len(cell_ids_all), len(genes_all)))
            spatial_all = None
            if "obsm" in handle and "spatial" in handle["obsm"]:
                spatial_all = np.asarray(handle["obsm"]["spatial"][:], dtype=float)
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise ValueError(f"Cannot read required H5AD fields from {path}: {exc}") from exc

    if matrix_all.shape != (len(cell_ids_all), len(genes_all)):
        raise ValueError(f"{path.name} X shape does not match obs/var identifiers")
    if len(regions_all) != len(cell_ids_all) or len(pixel_area_all) != len(cell_ids_all):
        raise ValueError(f"{path.name} obs fields have inconsistent lengths")
    if spatial_all is not None and spatial_all.shape[0] != len(cell_ids_all):
        raise ValueError(f"{path.name} obsm/spatial has inconsistent row count")
    if np.any(matrix_all.data < 0) or not np.isfinite(matrix_all.data).all():
        raise ValueError(f"{path.name} X must contain finite non-negative counts")

    keep_gene = genes_all != "blank"
    genes = genes_all[keep_gene]
    matrix = matrix_all[:, keep_gene].tocsr()
    total_counts = np.asarray(matrix.sum(axis=1)).ravel().astype(float)
    detected = _count_positive_per_row(matrix)
    qc_mask = (
        (total_counts > 4)
        & (total_counts < 250)
        & (detected > 3)
        & (pixel_area_all < 20000)
        & (pixel_area_all > 1000)
    )
    source_rows = np.flatnonzero(qc_mask)
    matrix = matrix[qc_mask].tocsr()
    matrix.data = np.log1p(matrix.data)
    regions = regions_all[qc_mask]
    if is_bad:
        regions = _replace_labels(regions, {"SPFp": "TH_SPFp"})
    region2 = _replace_labels(regions, {"IP": "CB_IP", "TH_ex": "TH", "other": "fiber"})
    region3 = _replace_labels(
        regions,
        {"IP": "CB", "TH_ex": "TH", "TH_RT": "TH", "TH_SPFp": "TH", "other": "fiber"},
    )
    display_regions = _replace_labels(regions, {"IP": "CB_IP", "TH_ex": "TH", "other": "fiber"})
    analysis_regions = _replace_labels(
        regions,
        {"IP": "CB", "TH_ex": "TH", "TH_RT": "TH", "TH_SPFp": "TH", "other": "fiber"},
    )
    unknown = sorted(set(display_regions) - set(DISPLAY_REGIONS))
    if unknown:
        raise ValueError(f"Unmapped display regions in {path.name}: {unknown}")
    return {
        "path": str(path),
        "genes": genes,
        "X": matrix,
        "regions": regions,
        "region2": region2,
        "region3": region3,
        "display_regions": display_regions,
        "analysis_regions": analysis_regions,
        "spatial": None if spatial_all is None else spatial_all[qc_mask],
        "cell_ids": cell_ids_all[qc_mask],
        "source_rows": source_rows,
        "pixel_area": pixel_area_all[qc_mask],
        "total_counts": total_counts[qc_mask],
        "n_genes_by_counts": detected[qc_mask],
        "n_before_qc": int(qc_mask.size),
        "n_after_qc": int(qc_mask.sum()),
        "n_dropped_qc": int((~qc_mask).sum()),
        "n_blank_vars": int((~keep_gene).sum()),
        "n_vars_after_blank": int(keep_gene.sum()),
        "n_explicit_zero_entries": int(np.count_nonzero(matrix.data == 0)),
    }


def load_allen_reference(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalise either the updated Allen table or its compact workbook sheet."""

    id_column = _id_column(frame, ("gene", "gene_symbol", "gene_name", "\u57fa\u56e0", "\u57fa\u56e0\u540d\u79f0"))
    genes = _clean_ids(frame[id_column], "Allen reference")
    raw = frame.copy()
    raw.index = genes
    if raw.index.has_duplicates:
        raise ValueError("Allen reference contains duplicated gene identifiers")
    raw = raw.drop(columns=[id_column])
    # A compact workbook can store the already-normalised eleven log2 columns.
    if set(REGIONS).issubset(raw.columns):
        # Keep workbook column order for deterministic tie handling in the
        # lowest-expression reference region. Correlations use REGIONS below.
        ordered_regions = [column for column in raw.columns if column in REGIONS]
        expr = raw[ordered_regions].apply(pd.to_numeric, errors="raise")
        expr.index = genes
        return expr

    p_pons_col = _find_column(raw, ("P_Pons_\u539f\u59cb\u8868\u8fbe\u91cf", "P_Pons_raw", "P_Pons"), required=False)
    my_medulla_col = _find_column(raw, ("MY_Medulla_\u539f\u59cb\u8868\u8fbe\u91cf", "MY_Medulla_raw", "MY_Medulla"), required=False)
    if p_pons_col is not None and my_medulla_col is not None:
        p_pons = pd.to_numeric(raw[p_pons_col], errors="coerce")
        my_medulla = pd.to_numeric(raw[my_medulla_col], errors="coerce")
        raw["HB"] = np.log2(pd.concat([p_pons, my_medulla], axis=1).mean(axis=1))
    log2_columns = [column for column in raw.columns if str(column).endswith("Log2\u503c")]
    if not log2_columns:
        log2_columns = [column for column in raw.columns if "log2" in _normalise_name(column)]
    expr = raw[log2_columns].apply(pd.to_numeric, errors="coerce")
    mapped: dict[str, str] = {}
    for column in expr.columns:
        normalised = _normalise_name(column)
        for region in REGIONS:
            if normalised.startswith(_normalise_name(region)) or normalised.startswith(_normalise_name(region) + "_"):
                mapped[column] = region
                break
    if "HB" in raw.columns:
        expr["HB"] = pd.to_numeric(raw["HB"], errors="coerce")
        mapped["HB"] = "HB"
    expr = expr.rename(columns=mapped)
    if not set(REGIONS).issubset(expr.columns):
        missing = sorted(set(REGIONS) - set(expr.columns))
        raise ValueError(f"Allen reference is missing normalized region columns: {missing}")
    expr = expr[[column for column in expr.columns if column in REGIONS]]
    if expr.isna().any().any():
        raise ValueError("Allen reference contains missing regional expression values")
    expr.index = genes
    return expr


def allen_region_dicts(allen: pd.DataFrame) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    top3: dict[str, list[str]] = {}
    wrong: dict[str, list[str]] = {}
    for gene, row in allen.iterrows():
        top3[str(gene)] = row.sort_values(ascending=False).head(3).index.tolist()
        wrong[str(gene)] = [str(row.idxmin())]
    return top3, wrong
