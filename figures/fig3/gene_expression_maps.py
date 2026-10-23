"""Load and render the final 14-gene spot cache as standalone rasters."""

from __future__ import annotations

try:
    from .data_io import SPATIAL_GENES, _read_json
except ImportError:  # direct import from this directory
    from data_io import SPATIAL_GENES, _read_json


GENE_COLORS = (
    "#FF3B30", "#FF9500", "#FFD60A", "#30D158", "#00C7BE", "#64D2FF",
    "#0A84FF", "#5856D6", "#AF52DE", "#FF2D55", "#FF375F", "#FF9F0A",
    "#5E5CE6", "#D1D1D6",
)


def load_spot_cache(ctx, genes=SPATIAL_GENES):
    """Read counts bounds and exactly the requested final gene caches."""

    np, _, _, _, _ = _deps()
    counts_path = ctx.locator.find(
        "spot_counts", ("data/spot_cache/counts.json", "spot_cache/counts.json", "counts.json"),
        required=True,
    )
    ctx._record_input("spot_counts", counts_path)
    bounds_data = _read_json(counts_path)
    bounds = bounds_data.get("bounds")
    if not bounds or len(bounds) != 4:
        raise ValueError(f"spot_cache/counts.json has no bounds=[xmin,ymin,xmax,ymax]: {counts_path}")
    spot_data = {}
    rows = []
    missing = []
    for gene in genes:
        path = ctx.locator.find(
            f"spot:{gene}", (f"data/spot_cache/{gene}.npy", f"spot_cache/{gene}.npy", f"{gene}.npy"),
            required=False,
        )
        if path is None:
            missing.append(gene)
            continue
        ctx._record_input(f"spot:{gene}", path)
        points = np.load(path, allow_pickle=False)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"{path} must be N x 2 [x_adjust,y_adjust], got {points.shape}")
        spot_data[gene] = points
        rows.append((gene, int(points.shape[0]), str(path), "x_adjust,y_adjust"))
    if missing:
        raise FileNotFoundError(f"Missing final 14-gene spot caches: {missing}")
    return [float(value) for value in bounds], spot_data, rows


def render_spatial_panel(path, spot_data, bounds, genes=SPATIAL_GENES):
    """Render one scientific 4 × 4 panel; no PDF/page composition."""

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    xmin, ymin, xmax, ymax = bounds
    fig, axes = plt.subplots(4, 4, figsize=(12, 9), facecolor="black")
    for ax in axes.flat:
        ax.set_facecolor("black")
        ax.set_xlim(xmax, xmin)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.axis("off")
    for ax, gene, color in zip(axes.flat, genes, GENE_COLORS):
        points = spot_data[gene]
        ax.scatter(points[:, 0], points[:, 1], s=.08, c=color,
                   linewidths=0, rasterized=True)
        ax.set_title(gene, color="white", fontsize=8, pad=2)
    for ax in axes.flat[len(genes):]:
        ax.axis("off")
    fig.tight_layout(pad=.3)
    fig.savefig(path, dpi=300, facecolor="black")
    plt.close(fig)
    return True


def _deps():
    try:
        from .data_io import _deps as shared_deps
    except ImportError:
        from data_io import _deps as shared_deps
    return shared_deps()

