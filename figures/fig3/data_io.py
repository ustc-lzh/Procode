"""Shared read-only inputs and lightweight analysis context for Fig. 3.

The workbook is a compact source-data manifest.  H5AD, CSV and NPY payloads
remain in the external ``--data-root`` tree and are never copied into this
code directory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .paths import DEFAULT_DATA_ROOT, DEFAULT_OUTPUT, DEFAULT_WORKBOOK
except ImportError:  # direct ``python module.py`` execution
    from paths import DEFAULT_DATA_ROOT, DEFAULT_OUTPUT, DEFAULT_WORKBOOK


STAGES = ("all", "celltypes", "markers", "correspondence", "correlation", "spatial")

# The final spatial panel.  Vip is an external legacy cache only and is not
# part of the current panel.
SPATIAL_GENES = (
    "Cux2", "Rorb", "Fezf2", "Cplx3", "Wfs1", "Prox1", "Drd1", "Prkcd",
    "Slc17a6", "Cbln1", "Gad1", "Calb2", "Mog", "Col1a1",
)

# Current source-data order for the 40-gene marker table.
MARKER_GENES = (
    "Cux2", "Rasgrf2", "Rorb", "Rspo1", "Satb2", "Gm19410", "Vwc2l", "Rprm",
    "Ccn2", "Cplx3", "Slc17a7", "Slc17a6", "Wfs1", "Necab2", "Prox1", "Prkcd",
    "Cbln1", "Drd1", "Drd2", "Pvalb", "Gad1", "Gad2", "Calb2", "Avp", "Oxt",
    "Chat", "Slc5a7", "Th", "Slc6a3", "Aqp4", "Mog", "Opalin", "Pdgfra", "Cspg4",
    "P2ry12", "Tmem119", "Cldn5", "Gucy1a1", "Spag16", "Col1a1",
)

# Canonical final 32 displayed groups.  Unresolved is kept separately and
# excluded from display statistics.
DISPLAY_GROUPS = (
    ("L2/3 IT-like", "L2/3 IT-like", "#FF4A3D"),
    ("L4/5 IT-like", "L4/5 IT-like", "#FF9430"),
    ("L5 IT-like", "L5 IT-like", "#FFE44A"),
    ("L5 ET-like", "L5 ET-like", "#FFBE3B"),
    ("L5/6 NP-like", "L5/6 NP-like", "#B7E85F"),
    ("L6 CT-like", "L6 CT-like", "#63D96B"),
    ("L6b-like", "L6b-like", "#20C997"),
    ("Cortex_HPF_Ex", "OLF_cortex_Ex", "#37D67A"),
    ("CA1", "CA1 pyramidal-like", "#00C7BE"),
    ("CA2", "CA2 neurons", "#30D5C8"),
    ("CA3", "CA3 pyramidal-like", "#5CE1D5"),
    ("HPF_DG", "DG granule neurons", "#64D2FF"),
    ("TH_neuron", "TH Glut neurons", "#5856D6"),
    ("MB_HB_HY_Glut", "MB_HB_HY_Glut", "#AF52DE"),
    ("CB_neuron", "CB granule-like", "#FF2D55"),
    ("STR_neuron", "Striatal SPNs", "#0A84FF"),
    ("TRN_GABA_neuron", "TRN GABAergic", "#6E6ADE"),
    ("MB_HB_HY_GABA", "MB_HB_HY_GABA", "#7A5195"),
    ("CB_GABA_neuron", "CB GABAergic neurons", "#FF6482"),
    ("OLF_In", "OLF GABA neurons", "#FF9F0A"),
    ("In", "Cortical/HPF GABA", "#FF375F"),
    ("HY_Avp_Oxt_like", "HY_Avp/Oxt-like", "#D78DE5"),
    ("Chat_neuron", "Cholinergic neurons", "#FF6B6B"),
    ("MB_dopa", "MB dopaminergic", "#BF5AF2"),
    ("Astro", "Astrocytes", "#34C759"),
    ("Oligo", "Oligodendrocytes", "#5E5CE6"),
    ("OPC", "OPCs", "#7D7AFF"),
    ("Microglia", "Microglia", "#8E8E93"),
    ("Endothelial", "Endothelial cells", "#A7F3D0"),
    ("Vascular_mural", "Vascular mural cells", "#C7A17A"),
    ("Ependymal", "Ependymal cells", "#40C8E0"),
    ("Meningeal", "Meningeal cells", "#D1D1D6"),
)

DEFAULT_CORTICAL_MAPPING = (
    ("C0", "L2/3 IT-like", "retain as class", "canonical marker combination and spatial band"),
    ("C1", "L5 IT-like", "retain as class", "canonical marker combination and spatial band"),
    ("C2", "L6 CT-like", "retain as class", "canonical marker combination and spatial band"),
    ("C3", "L4/5 IT-like", "retain as class", "canonical marker combination and spatial band"),
    ("C4", "L5 ET-like", "retain as class", "canonical marker combination and spatial band"),
    ("C5", "L6b-like", "retain as class", "canonical marker combination and spatial band"),
    ("C6", "L5/6 NP-like", "retain as class", "canonical marker combination and spatial band"),
    ("C7", "L2/3 IT-like", "merge into C0 L2/3 IT-like class", "Cux2/Rasgrf2 superficial band; low count"),
    ("C8", "L5 ET-like", "merge into C4 ET-like class", "Fezf2/Bcl11b/Gm19410 and deep position"),
)

CORRESPONDENCE_MANUAL_ORDER = (
    "CB", "HPF", "Isocortex", "CTXsp", "OLF", "STR", "VL", "fiber",
    "TH", "TH_RT", "TH_SPFp", "HB", "HY", "MB", "PAL", "CB_IP",
)
CORRESPONDENCE_DOMAIN_ORDER = (
    "CB_unknown", "GCL", "ML", "PCL", "CA1", "CA3", "DG", "HPF",
    "L2-4", "L5", "L6 CT", "L6b", "OLF", "STR", "VS", "Meninges",
    "WM", "TH", "HY_MB_HB",
)
MANUAL_MERGE = {
    "Isocortex": "Isocortex", "CTXsp": "Isocortex", "TH": "TH",
    "TH_RT": "HY_MB_HB", "TH_SPFp": "HY_MB_HB", "HB": "HY_MB_HB",
    "HY": "HY_MB_HB", "MB": "HY_MB_HB", "PAL": "HY_MB_HB",
    "CB_IP": "HY_MB_HB", "CB": "CB", "HPF": "HPF", "OLF": "OLF",
    "STR": "STR", "VL": "VL", "fiber": "fiber",
}
DOMAIN_MERGE = {
    "CB_unknown": "CB", "GCL": "CB", "ML": "CB", "PCL": "CB",
    "CA1": "HPF", "CA3": "HPF", "DG": "HPF", "HPF": "HPF",
    "L2-4": "Isocortex", "L5": "Isocortex", "L6 CT": "Isocortex",
    "L6b": "Isocortex", "Meninges": "fiber", "WM": "fiber",
    "HY_MB_HB": "HY_MB_HB", "OLF": "OLF", "STR": "STR", "VS": "VS",
    "TH": "TH",
}
MERGED_ROWS = ("CB", "HPF", "Isocortex", "OLF", "STR", "VL", "fiber", "TH", "HY_MB_HB")
MERGED_COLUMNS = ("CB", "HPF", "Isocortex", "OLF", "STR", "VS", "fiber", "TH", "HY_MB_HB")


class MissingInput(FileNotFoundError):
    """Raised when a requested stage lacks a required input."""


class DataLocator:
    """Resolve release-relative files below the explicit external data root."""

    def __init__(self, root: Path):
        self.root = root.expanduser()

    def find(self, logical: str, defaults: Sequence[str], required: bool = True) -> Path | None:
        candidates: list[Path] = []
        for raw in defaults:
            path = Path(raw).expanduser()
            if path.is_absolute():
                candidates.append(path)
            else:
                candidates.append(self.root / path)
                # Support extracting the supplied archive one directory above
                # its Result3_data_20261020 payload, without any old workdir.
                candidates.append(self.root / "Result3_data_20261020" / path)
        seen: set[str] = set()
        for path in candidates:
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            if path.is_file():
                return path
        if required:
            raise MissingInput(f"{logical}: no file found below {self.root}; tried {seen}")
        return None


def _deps() -> tuple[Any, Any, Any, Any, Any]:
    """Load scientific dependencies lazily so ``--help`` stays lightweight."""

    try:
        import h5py  # type: ignore
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore
        from scipy import sparse, stats  # type: ignore
        return np, pd, h5py, sparse, stats
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "Fig. 3 analysis needs numpy, pandas, h5py and scipy. "
            "Use the project runtime/PYTHONPATH described in README.md."
        ) from exc


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return value


def _decode(values: Any) -> list[str]:
    result: list[str] = []
    for value in values:
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        result.append(str(value))
    return result


def _h5_vector(group: Any, key: str, decode: bool = True) -> Any:
    obj = group[key]
    if hasattr(obj, "keys"):
        categories = _decode(obj["categories"][:])
        codes = obj["codes"][:]
        return ["" if int(code) < 0 else categories[int(code)] for code in codes]
    values = obj[:]
    return _decode(values) if decode else values


def _h5_index(group: Any) -> list[str]:
    key = group.attrs.get("_index", "_index")
    if isinstance(key, bytes):
        key = key.decode("utf-8")
    return _decode(group[str(key)][:])


def _read_h5ad(path: Path, *, need_raw: bool = False, need_umap: bool = False,
               obs_columns: Sequence[str] = ()) -> dict[str, Any]:
    np, _, h5py, sparse, _ = _deps()
    out: dict[str, Any] = {}
    with h5py.File(path, "r", locking=False) as handle:
        out["ids"] = np.asarray(_h5_index(handle["obs"]), dtype=object)
        if "spatial" in handle.get("obsm", {}):
            out["spatial"] = handle["obsm"]["spatial"][:]
        if need_umap and "X_umap" in handle.get("obsm", {}):
            out["umap"] = handle["obsm"]["X_umap"][:]
        out["genes"] = np.asarray(_h5_index(handle["var"]), dtype=object)
        for column in obs_columns:
            if column in handle["obs"]:
                out[column] = np.asarray(_h5_vector(handle["obs"], column, decode=False))
        if need_raw:
            group = handle["layers"]["raw"]
            shape = tuple(int(x) for x in group.attrs["shape"])
            out["raw"] = sparse.csr_matrix(
                (group["data"][:], group["indices"][:], group["indptr"][:]), shape=shape
            )
    return out


def _read_csv(path: Path, **kwargs: Any) -> Any:
    _, pd, _, _, _ = _deps()
    return pd.read_csv(path, **kwargs)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_workbook(path: Path) -> dict[str, Any]:
    """Read the supplied workbook without modifying or re-saving it."""

    if not path.is_file():
        raise MissingInput(f"workbook: file not found: {path}")
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"Workbook JSON must be an object: {path}")
        return value
    if path.suffix.lower() in {".xlsx", ".xls"}:
        try:
            _, pd, _, _, _ = _deps()
            sheets = []
            for name in pd.ExcelFile(path).sheet_names:
                frame = pd.read_excel(path, sheet_name=name)
                frame = frame.where(frame.notna(), None)
                sheets.append({
                    "name": str(name), "source": str(path),
                    "description": "read-only workbook input", "role": "input",
                    "required": False, "columns": [str(x) for x in frame.columns],
                    "rows": frame.astype(object).values.tolist(),
                })
            return {"figure": "fig3", "sheets": sheets, "external_files": [], "notes": []}
        except Exception as exc:
            raise RuntimeError(f"Could not read workbook {path}: {exc}") from exc
    raise ValueError(f"Unsupported workbook format: {path}")


def _sheet_rows(workbook: Mapping[str, Any], name: str) -> list[dict[str, Any]]:
    for sheet in workbook.get("sheets", []):
        if str(sheet.get("name", "")) != name:
            continue
        columns = list(sheet.get("columns", []))
        rows = []
        for raw in sheet.get("rows", []):
            if isinstance(raw, Mapping):
                rows.append(dict(raw))
            else:
                rows.append(dict(zip(columns, raw)))
        return rows
    return []


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


class AnalysisContext:
    """Shared workbook configuration, external locator and report state."""

    def __init__(self, locator: DataLocator, workbook: dict[str, Any], output: Path):
        self.locator = locator
        self.workbook = workbook
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        self.data: dict[str, Any] | None = None
        self.annotation: Any | None = None
        self.config = self._read_config()
        self.report: dict[str, Any] = {
            "figure": "fig3", "analysis": "compact_fig3_analysis",
            "stage": {}, "inputs": {}, "checks": {}, "formulas": {},
            "unresolved": [], "outputs": {},
        }

    def _read_config(self) -> dict[str, Any]:
        mapping_rows = _sheet_rows(self.workbook, "cortical_mapping")
        if mapping_rows:
            mapping = []
            for row in mapping_rows:
                cluster = str(row.get("original_r40_cluster", row.get("cluster", "")))
                label = str(row.get("recommended_label", row.get("label", "")))
                if cluster and label:
                    mapping.append((cluster, label, str(row.get("action", "")), str(row.get("reason", ""))))
        else:
            mapping = list(DEFAULT_CORTICAL_MAPPING)

        group_rows = _sheet_rows(self.workbook, "display_groups")
        if group_rows:
            groups = []
            for row in group_rows:
                key = str(row.get("label_key", row.get("figure_display_label_key", "")))
                name = str(row.get("display_name", row.get("figure_display_name", key)))
                color = str(row.get("color", "#808080"))
                if key and key != "Unresolved":
                    groups.append((key, name, color))
        else:
            groups = list(DISPLAY_GROUPS)

        marker_rows = _sheet_rows(self.workbook, "markers")
        marker_genes = []
        for row in marker_rows:
            gene = str(row.get("gene", "")).strip()
            if gene and gene not in marker_genes:
                marker_genes.append(gene)
        if not marker_genes:
            marker_genes = list(MARKER_GENES)
        return {"mapping": mapping, "groups": groups, "marker_genes": marker_genes}

    @property
    def groups(self) -> list[tuple[str, str, str]]:
        return self.config["groups"]

    @property
    def group_keys(self) -> list[str]:
        return [item[0] for item in self.groups]

    def _record_input(self, logical: str, path: Path | None) -> None:
        if path is not None:
            self.report["inputs"][logical] = str(path)

    def load_main(self) -> dict[str, Any]:
        if self.data is not None:
            return self.data
        h5ad = self.locator.find(
            "fig3_refined", ("data/fig3_refined.h5ad", "fig3_refined.h5ad"), required=True
        )
        self._record_input("fig3_refined", h5ad)
        self.data = _read_h5ad(
            h5ad, need_raw=True, need_umap=True,
            obs_columns=("n_genes_by_counts", "total_counts"),
        )
        ann_path = self.locator.find(
            "figure_display_annotations", (
                "source_data/figure_display_annotations.csv",
                "figure_display_annotations.csv",
            ), required=True,
        )
        self._record_input("figure_display_annotations", ann_path)
        ann = _read_csv(ann_path, dtype=str, keep_default_na=False)
        ids = self.data["ids"].astype(str)
        if "cell_id" not in ann.columns:
            raise ValueError(f"Annotation table has no cell_id column: {ann_path}")
        ann["cell_id"] = ann["cell_id"].astype(str)
        if not ann["cell_id"].is_unique:
            raise ValueError("Final display annotation cell_id values are not unique")
        ann = ann.set_index("cell_id").reindex(ids)
        if ann.index.isna().any() or ann.isna().all(axis=1).any():
            raise ValueError("Final display annotation table does not cover H5AD cell IDs")
        if "figure_display_label_key" not in ann.columns:
            source = "final_label_key" if "final_label_key" in ann.columns else "old_label_key"
            ann["figure_display_label_key"] = ann[source]
        if "figure_display_name" not in ann.columns:
            ann["figure_display_name"] = ann["figure_display_label_key"]
        if "shown_in_figures" not in ann.columns:
            ann["shown_in_figures"] = ann["figure_display_label_key"].ne("Unresolved")
        if "non_iso_reassigned_by_existing_cluster" not in ann.columns:
            ann["non_iso_reassigned_by_existing_cluster"] = False
        self.annotation = ann.reset_index(names="cell_id")
        self.data["labels"] = self.annotation["figure_display_label_key"].to_numpy(dtype=object)
        self.data["names"] = self.annotation["figure_display_name"].to_numpy(dtype=object)
        self.data["shown"] = self.annotation["shown_in_figures"].map(_as_bool).to_numpy()
        self.report["checks"].update({
            "main_cells": int(len(ids)),
            "annotation_cells": int(len(self.annotation)),
            "annotation_order_matches_h5ad": True,
            "display_groups_observed": sorted(set(self.data["labels"])),
            "unresolved_cells": int((self.data["labels"] == "Unresolved").sum()),
            "non_iso_reassigned_cells": int(self.annotation["non_iso_reassigned_by_existing_cluster"].map(_as_bool).sum()),
            "displayed_cells": int(self.data["shown"].sum()),
        })
        self._record_input("cell_assignments_exploratory", self.locator.find(
            "cell_assignments_exploratory", (
                "data/cell_assignments_exploratory.csv", "cell_assignments_exploratory.csv",
            ), required=False,
        ))
        return self.data


def _save_table(path: Path, frame: Any) -> None:
    frame.to_csv(path, index=False)

