"""Marker normalization/statistics for the 32 displayed populations."""

from __future__ import annotations

try:
    from .data_io import MARKER_GENES, _deps, _save_table
    from .supplementary_analysis import render_marker_dotplot, render_qc
except ImportError:  # direct import from this directory
    from data_io import MARKER_GENES, _deps, _save_table
    from supplementary_analysis import render_marker_dotplot, render_qc


def marker_genes(ctx) -> list[str]:
    genes = [str(value) for value in ctx.config["marker_genes"] if str(value)]
    if len(genes) != len(set(genes)):
        raise ValueError("Marker workbook contains duplicate genes")
    return genes


def compute_marker_statistics(ctx, genes):
    """Return frame, mean matrix, detection matrix and scaled means."""

    np, pd, _, _, _ = _deps()
    data = ctx.load_main()
    gene_lookup = {str(gene): i for i, gene in enumerate(data["genes"])}
    missing = [gene for gene in genes if gene not in gene_lookup]
    if missing:
        raise ValueError(f"Marker genes absent from fig3_refined.h5ad: {missing}")
    raw = data["raw"]
    selected = raw[:, [gene_lookup[gene] for gene in genes]].tocsr()
    totals = np.asarray(raw.sum(axis=1)).ravel().astype(float)
    # The denominator is the full 304-gene raw library, not marker-only counts.
    normalized = selected.multiply((500.0 / np.maximum(totals, 1.0))[:, None]).tocsr().astype(float)
    normalized.data = np.log1p(normalized.data)
    means = []
    fractions = []
    labels = data["labels"]
    for key in ctx.group_keys:
        mask = labels == key
        means.append(np.asarray(normalized[mask].mean(axis=0)).ravel())
        fractions.append(np.asarray((selected[mask] > 0).sum(axis=0)).ravel() / max(int(mask.sum()), 1))
    mean_matrix = np.asarray(means, dtype=float)
    fraction_matrix = np.asarray(fractions, dtype=float)
    lo = mean_matrix.min(axis=0)
    hi = mean_matrix.max(axis=0)
    scaled = np.divide(mean_matrix - lo, hi - lo, out=np.zeros_like(mean_matrix), where=hi > lo)
    records = []
    for i, (key, name, _) in enumerate(ctx.groups):
        for j, gene in enumerate(genes):
            records.append((key, name, gene, int((labels == key).sum()),
                            float(mean_matrix[i, j]), float(fraction_matrix[i, j]),
                            float(scaled[i, j])))
    frame = pd.DataFrame(records, columns=[
        "label_key", "display_label", "gene", "n_cells", "mean_log1p_cp500",
        "fraction_detected_raw_gt0", "per_gene_scaled_mean",
    ])
    return frame, mean_matrix, fraction_matrix, scaled


def run_markers(ctx) -> None:
    np, pd, _, _, _ = _deps()
    data = ctx.load_main()
    genes = marker_genes(ctx)
    frame, means, fractions, scaled = compute_marker_statistics(ctx, genes)
    _save_table(ctx.output / "marker_statistics.csv", frame)
    ctx.report["formulas"].update({
        "marker_mean": "mean(log1p(raw / full_304_gene_library_size * 500))",
        "marker_detection": "fraction(raw > 0)",
        "marker_scaling": "per-gene min-max across the 32 displayed groups",
    })
    ctx.report["checks"]["marker_rows"] = int(len(frame))
    ctx.report["checks"]["marker_genes"] = genes

    reference = ctx.locator.find(
        "marker_statistics", (
            "source_data/SFig5_C_marker_statistics.csv", "SFig5_C_marker_statistics.csv",
        ), required=False,
    )
    if reference is not None:
        ctx._record_input("marker_statistics", reference)
        ref = pd.read_csv(reference)
        if all(column in ref.columns for column in ("label_key", "gene")):
            compare = frame.merge(ref, on=["label_key", "gene"], suffixes=("_computed", "_source"))
            diff = {}
            for column in ("mean_log1p_cp500", "fraction_detected_raw_gt0", "per_gene_scaled_mean"):
                left, right = f"{column}_computed", f"{column}_source"
                if left in compare and right in compare:
                    diff[column] = (float(np.nanmax(np.abs(compare[left].to_numpy(float) -
                                                       compare[right].to_numpy(float)))) if len(compare) else None)
            ctx.report["checks"]["marker_source_rows"] = int(len(ref))
            ctx.report["checks"]["marker_source_max_abs_diff"] = diff
            ctx.report["checks"]["marker_source_match"] = bool(
                diff and all(value is not None and value < 1e-8 for value in diff.values())
            )

    render_qc(ctx, data)
    render_marker_dotplot(ctx, genes, ctx.group_keys, fractions, scaled)
    ctx.report["outputs"]["marker_statistics"] = str(ctx.output / "marker_statistics.csv")
    ctx.report["stage"]["markers"] = "complete"

