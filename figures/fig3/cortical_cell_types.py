"""Check cortical cluster assignments against the annotated cell types."""

from __future__ import annotations

try:
    from .data_io import _deps
except ImportError:  # direct import from this directory
    from data_io import _deps


def validate_cortical_mapping(ctx) -> dict[str, object]:
    """Compare recorded r40 assignments with final display labels when present."""

    np, pd, _, _, _ = _deps()
    mapping = ctx.config["mapping"]
    mapping_dict = {str(row[0]): str(row[1]) for row in mapping}
    path = ctx.locator.find(
        "cell_assignments_exploratory", (
            "data/cell_assignments_exploratory.csv", "cell_assignments_exploratory.csv",
        ), required=False,
    )
    if path is None:
        return {"available": False, "reason": "cell_assignments_exploratory.csv not provided"}
    ctx._record_input("cell_assignments_exploratory", path)
    table = pd.read_csv(path, dtype={"cell_id": str, "leiden_r40": str})
    if "leiden_r40" not in table.columns:
        return {"available": False, "reason": "leiden_r40 column not provided"}
    labels = ctx.annotation.set_index("cell_id")["figure_display_label_key"]
    table["cell_id"] = table["cell_id"].astype(str)
    # Exploratory CSV values are bare integers; the workbook uses C0..C8.
    table["cluster_key"] = "C" + table["leiden_r40"].astype(str).str.replace(r"^C", "", regex=True)
    table["expected_display_label"] = table["cluster_key"].map(mapping_dict)
    table["actual_display_label"] = table["cell_id"].map(labels)
    covered = table["expected_display_label"].notna() & table["actual_display_label"].notna()
    mismatches = table.loc[covered, "expected_display_label"].ne(table.loc[covered, "actual_display_label"])
    cortical = table.loc[covered]
    ctx.report["checks"]["cortical_cells_covered"] = int(len(cortical))
    ctx.report["checks"]["cortical_mapping_mismatches"] = int(mismatches.sum())
    return {
        "available": True, "n_rows": int(len(table)), "n_covered": int(covered.sum()),
        "mismatches": int(mismatches.sum()),
        "expected_classes": sorted(set(mapping_dict.values())),
        "cluster_counts": table["leiden_r40"].value_counts().sort_index().to_dict(),
    }

