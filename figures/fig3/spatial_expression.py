"""Final spatial spot-cache stage for Fig. 3."""

from __future__ import annotations

try:
    from .data_io import SPATIAL_GENES, _deps, _sheet_rows
    from .gene_expression_maps import load_spot_cache, render_spatial_panel
except ImportError:  # direct import from this directory
    from data_io import SPATIAL_GENES, _deps, _sheet_rows
    from gene_expression_maps import load_spot_cache, render_spatial_panel


def run_spatial(ctx) -> None:
    np, pd, _, _, _ = _deps()
    bounds, spot_data, rows = load_spot_cache(ctx, SPATIAL_GENES)
    expected_rows = {
        str(row.get("gene")): int(float(row.get("n_spots")))
        for row in _sheet_rows(ctx.workbook, "spot_counts")
        if row.get("gene") and row.get("n_spots") not in (None, "")
    }
    if expected_rows:
        ctx.report["checks"]["spot_counts_sheet_match"] = (
            all(expected_rows.get(gene) == int(points.shape[0]) for gene, points in spot_data.items())
            and set(expected_rows) == set(spot_data)
        )
    pd.DataFrame(rows, columns=["gene", "n_spots", "source", "columns"]).to_csv(
        ctx.output / "spatial_gene_counts.csv", index=False
    )
    ctx.report["formulas"]["spatial_coordinates"] = (
        "cached spot coordinates [x_adjust, y_adjust]; shared bounds from counts.json"
    )
    ctx.report["checks"].update({
        "spatial_gene_order": list(SPATIAL_GENES),
        "spatial_bounds": bounds,
        "legacy_vip_excluded": "Vip" not in SPATIAL_GENES,
    })
    ctx.report["outputs"]["spatial_gene_counts"] = str(ctx.output / "spatial_gene_counts.csv")
    panel = ctx.output / "spatial_gene_panel.png"
    if render_spatial_panel(panel, spot_data, bounds, SPATIAL_GENES):
        ctx.report["outputs"]["spatial_gene_panel"] = str(panel)
    else:
        ctx.report["unresolved"].append("matplotlib unavailable; spatial panel not rendered")
    ctx.report["stage"]["spatial"] = "complete"

