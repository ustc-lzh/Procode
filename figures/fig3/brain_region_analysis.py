"""Correspondence and positive-cell correlation analyses for Fig. 3."""

from __future__ import annotations

try:
    from .data_io import (
        CORRESPONDENCE_DOMAIN_ORDER,
        CORRESPONDENCE_MANUAL_ORDER,
        DOMAIN_MERGE,
        MANUAL_MERGE,
        MERGED_COLUMNS,
        MERGED_ROWS,
        _deps,
        _read_csv,
        _read_h5ad,
        _sheet_rows,
        _save_table,
    )
except ImportError:  # direct import from this directory
    from data_io import (
        CORRESPONDENCE_DOMAIN_ORDER,
        CORRESPONDENCE_MANUAL_ORDER,
        DOMAIN_MERGE,
        MANUAL_MERGE,
        MERGED_COLUMNS,
        MERGED_ROWS,
        _deps,
        _read_csv,
        _read_h5ad,
        _sheet_rows,
        _save_table,
    )


def matrix_from_frame(frame, rows, columns):
    """Align brain-region counts to the specified row and column order."""

    _, _, _, _, _ = _deps()
    frame = frame.copy()
    if frame.empty or len(frame.columns) == 0:
        raise ValueError("Correspondence count table is empty")
    first = frame.columns[0]
    frame = frame.set_index(first)
    frame.index = frame.index.astype(str)
    frame.columns = frame.columns.astype(str)
    frame = frame.reindex(index=list(rows), columns=list(columns), fill_value=0)
    return frame.apply(lambda column: column.astype(float)).to_numpy()


def _compute_correspondence_from_h5ad(ctx):
    np, pd, _, _, _ = _deps()
    ref_path = ctx.locator.find(
        "fig3_reference", ("data/fig3_reference.h5ad", "fig3_reference.h5ad"), required=False
    )
    cast_path = ctx.locator.find(
        "adata_cast", ("data/adata_cast.h5ad", "adata_cast.h5ad"), required=False
    )
    if ref_path is None or cast_path is None:
        return None
    ctx._record_input("fig3_reference", ref_path)
    ctx._record_input("adata_cast", cast_path)
    ref = _read_h5ad(ref_path, obs_columns=("brain_region2",))
    cast = _read_h5ad(cast_path, obs_columns=("refined_domain",))
    common, ref_pos, cast_pos = np.intersect1d(ref["ids"], cast["ids"], return_indices=True)
    manual = np.asarray(ref["brain_region2"], dtype=object)[ref_pos]
    domains = np.asarray(cast["refined_domain"], dtype=object)[cast_pos]
    valid = np.isin(manual, CORRESPONDENCE_MANUAL_ORDER) & np.isin(domains, CORRESPONDENCE_DOMAIN_ORDER)
    ids = common[valid]
    manual = manual[valid]
    domains = domains[valid]
    matrix = np.zeros((len(CORRESPONDENCE_MANUAL_ORDER), len(CORRESPONDENCE_DOMAIN_ORDER)), dtype=np.int64)
    row_lookup = {key: i for i, key in enumerate(CORRESPONDENCE_MANUAL_ORDER)}
    col_lookup = {key: i for i, key in enumerate(CORRESPONDENCE_DOMAIN_ORDER)}
    for row, column in zip(manual, domains):
        matrix[row_lookup[str(row)], col_lookup[str(column)]] += 1
    normalized = np.divide(
        matrix, matrix.sum(axis=1, keepdims=True), out=np.zeros_like(matrix, dtype=float),
        where=matrix.sum(axis=1, keepdims=True) > 0,
    )
    coordinate_report = {}
    if "spatial" in ref and "spatial" in cast:
        delta = np.linalg.norm(ref["spatial"][ref_pos] - cast["spatial"][cast_pos], axis=1)
        coordinate_report = {"max": float(np.max(delta)), "median": float(np.median(delta))}
    common_frame = pd.DataFrame({
        "cell_id": ids, "manual_brain_region": manual, "cast_refined_domain": domains,
    })
    return matrix, normalized, common_frame, {
        "source": "h5ad_intersection",
        "n_common_valid_cells": int(len(ids)),
        "coordinate_delta": coordinate_report,
    }


def _load_precomputed_correspondence(ctx):
    np, pd, _, _, _ = _deps()
    counts_rows = _sheet_rows(ctx.workbook, "correspondence_counts")
    if counts_rows:
        counts_frame = pd.DataFrame(counts_rows)
        source = "workbook:correspondence_counts"
    else:
        counts_path = ctx.locator.find(
            "correspondence_counts", (
                "source_data/F_manual_vs_cast_unmerged_counts.csv",
                "tables/F_manual_vs_cast_unmerged_counts.csv",
            ), required=True,
        )
        ctx._record_input("correspondence_counts", counts_path)
        counts_frame = _read_csv(counts_path)
        source = str(counts_path)
    matrix = matrix_from_frame(
        counts_frame, CORRESPONDENCE_MANUAL_ORDER, CORRESPONDENCE_DOMAIN_ORDER
    ).astype(np.int64)
    normalized = np.divide(
        matrix, matrix.sum(axis=1, keepdims=True), out=np.zeros_like(matrix, dtype=float),
        where=matrix.sum(axis=1, keepdims=True) > 0,
    )
    common_path = ctx.locator.find(
        "correspondence_assignments", (
            "source_data/F_common_cell_assignments_unmerged.csv",
            "tables/F_common_cell_assignments_unmerged.csv",
        ), required=False,
    )
    if common_path is not None:
        ctx._record_input("correspondence_assignments", common_path)
        common = _read_csv(common_path, dtype={"cell_id": str})
    else:
        common = pd.DataFrame(columns=["cell_id", "manual_brain_region", "cast_refined_domain"])
    return matrix, normalized, common, {
        "source": source,
        # This is the number represented by the precomputed matrix.  It is
        # deliberately separate from the optional assignment-row count.
        "n_common_valid_cells": int(matrix.sum()),
        "assignment_rows_available": int(len(common)),
    }


def run_correspondence(ctx) -> None:
    np, pd, _, _, _ = _deps()
    computed = _compute_correspondence_from_h5ad(ctx)
    if computed is None:
        matrix, normalized, common, provenance = _load_precomputed_correspondence(ctx)
    else:
        matrix, normalized, common, provenance = computed
    pd.DataFrame(matrix, index=CORRESPONDENCE_MANUAL_ORDER,
                 columns=CORRESPONDENCE_DOMAIN_ORDER).to_csv(
                     ctx.output / "correspondence_counts.csv", index_label="manual_brain_region"
                 )
    pd.DataFrame(normalized, index=CORRESPONDENCE_MANUAL_ORDER,
                 columns=CORRESPONDENCE_DOMAIN_ORDER).to_csv(
                     ctx.output / "correspondence_row_normalized.csv", index_label="manual_brain_region"
                 )
    # Empty common assignments are valid for XLSX-only runs and must not
    # generate a misleading empty output artifact.
    if not common.empty:
        common.to_csv(ctx.output / "correspondence_assignments.csv", index=False)

    merged = np.zeros((len(MERGED_ROWS), len(MERGED_COLUMNS)), dtype=np.int64)
    row_lookup = {key: i for i, key in enumerate(MERGED_ROWS)}
    col_lookup = {key: i for i, key in enumerate(MERGED_COLUMNS)}
    for i, row in enumerate(CORRESPONDENCE_MANUAL_ORDER):
        for j, column in enumerate(CORRESPONDENCE_DOMAIN_ORDER):
            merged[row_lookup[MANUAL_MERGE[row]], col_lookup[DOMAIN_MERGE[column]]] += int(matrix[i, j])
    pd.DataFrame(merged, index=MERGED_ROWS, columns=MERGED_COLUMNS).to_csv(
        ctx.output / "correspondence_merged_counts.csv", index_label="manual_region_merged"
    )
    merged_norm = np.divide(
        merged, merged.sum(axis=1, keepdims=True), out=np.zeros_like(merged, dtype=float),
        where=merged.sum(axis=1, keepdims=True) > 0,
    )
    pd.DataFrame(merged_norm, index=MERGED_ROWS, columns=MERGED_COLUMNS).to_csv(
        ctx.output / "correspondence_merged_row_normalized.csv", index_label="manual_region_merged"
    )
    ctx.report["formulas"]["correspondence_row_normalized"] = "count / row sum; rows with zero total remain zero"
    ctx.report["checks"].update({
        "correspondence_shape": [int(x) for x in matrix.shape],
        "correspondence_total": int(matrix.sum()),
        "correspondence_row_sums_one": bool(
            np.allclose(normalized.sum(axis=1)[matrix.sum(axis=1) > 0], 1.0)
        ),
        "correspondence_provenance": provenance,
    })
    ctx.report["outputs"].update({
        "correspondence_counts": str(ctx.output / "correspondence_counts.csv"),
        "correspondence_row_normalized": str(ctx.output / "correspondence_row_normalized.csv"),
        "correspondence_merged_counts": str(ctx.output / "correspondence_merged_counts.csv"),
        "correspondence_merged_row_normalized": str(ctx.output / "correspondence_merged_row_normalized.csv"),
    })
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(11.0, 5.0), facecolor="white")
        image = ax.imshow(normalized, cmap="Reds", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(CORRESPONDENCE_DOMAIN_ORDER)), CORRESPONDENCE_DOMAIN_ORDER,
                      rotation=60, ha="right")
        ax.set_yticks(range(len(CORRESPONDENCE_MANUAL_ORDER)), CORRESPONDENCE_MANUAL_ORDER)
        ax.set_xlabel("CAST refined domain")
        ax.set_ylabel("Manual brain region")
        ax.set_title("Row-normalized correspondence")
        fig.colorbar(image, ax=ax, label="Row fraction")
        fig.tight_layout()
        path = ctx.output / "correspondence_heatmap.png"
        fig.savefig(path, dpi=300, facecolor="white")
        plt.close(fig)
        ctx.report["outputs"]["correspondence_heatmap"] = str(path)
    except ImportError:
        ctx.report["unresolved"].append("matplotlib unavailable; correspondence heatmap not rendered")
    ctx.report["stage"]["correspondence"] = "complete"


def run_correlation(ctx) -> None:
    np, pd, _, _, stats = _deps()
    path = ctx.locator.find(
        "correlation_points", (
            "data/tables/D_positive_cell_mean_correlation.csv",
            "tables/D_positive_cell_mean_correlation.csv",
            "D_positive_cell_mean_correlation.csv",
        ), required=False,
    )
    if path is None:
        rows = _sheet_rows(ctx.workbook, "correlation_points")
        if not rows:
            raise FileNotFoundError(
                "correlation_points: D_positive_cell_mean_correlation.csv not found and workbook sheet is absent"
            )
        data = pd.DataFrame(rows)
        source = "workbook:correlation_points"
    else:
        ctx._record_input("correlation_points", path)
        data = _read_csv(path)
        source = str(path)
    required = {"scrna_positive_mean", "spatial_positive_mean"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Correlation table missing columns: {sorted(missing)}")
    data.to_csv(ctx.output / "correlation_points.csv", index=False)
    x = data["scrna_positive_mean"].to_numpy(float)
    y = data["spatial_positive_mean"].to_numpy(float)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if valid.sum() < 2:
        raise ValueError("Correlation table has fewer than two positive finite points")
    pearson_r, pearson_p = stats.pearsonr(x[valid], y[valid])
    ctx.report["formulas"]["correlation_positive_filter"] = "finite x and y with x > 0 and y > 0"
    ctx.report["checks"].update({
        "correlation_source": source,
        "correlation_rows": int(len(data)),
        "correlation_positive_points": int(valid.sum()),
        "correlation_pearson_r": float(pearson_r),
        "correlation_pearson_p": float(pearson_p),
    })
    ctx.report["outputs"]["correlation_points"] = str(ctx.output / "correlation_points.csv")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5.2, 4.2), facecolor="white")
        ax.scatter(x[valid], y[valid], s=8, color="#2C7FB8", alpha=.75,
                   linewidths=0, rasterized=True)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("scRNA-seq positive-cell mean")
        ax.set_ylabel("Spatial positive-cell mean")
        ax.set_title(f"Positive-cell means (Pearson r = {pearson_r:.3f})")
        fig.tight_layout()
        path = ctx.output / "positive_mean_correlation.png"
        fig.savefig(path, dpi=300, facecolor="white")
        plt.close(fig)
        ctx.report["outputs"]["positive_mean_correlation"] = str(path)
    except ImportError:
        ctx.report["unresolved"].append("matplotlib unavailable; correlation scatter not rendered")
    ctx.report["stage"]["correlation"] = "complete"

