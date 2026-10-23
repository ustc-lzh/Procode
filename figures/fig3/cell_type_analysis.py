"""Cell-type counts, cortical provenance and standalone cell plots."""

from __future__ import annotations

try:
    from .cortical_cell_types import validate_cortical_mapping
    from .data_io import DEFAULT_CORTICAL_MAPPING, _as_bool, _deps, _save_table
except ImportError:  # direct script execution
    from cortical_cell_types import validate_cortical_mapping
    from data_io import DEFAULT_CORTICAL_MAPPING, _as_bool, _deps, _save_table


# Export the recorded mapping under the historical module name as well.  The
# final annotation labels remain authoritative; this is provenance only.
CORTICAL_MAPPING = DEFAULT_CORTICAL_MAPPING


def run_celltypes(ctx) -> None:
    """Write 32 shown groups plus Unresolved and optional scientific rasters."""

    np, pd, _, _, _ = _deps()
    data = ctx.load_main()
    labels = data["labels"]
    groups = ctx.groups
    rows = []
    for order, (key, name, color) in enumerate(groups):
        rows.append((key, name, color, order, int(np.sum(labels == key))))
    rows.append(("Unresolved", "Unresolved", "#BDBDBD", len(groups), int(np.sum(labels == "Unresolved"))))
    counts = pd.DataFrame(rows, columns=["label_key", "display_name", "color", "display_order", "n_cells"])
    _save_table(ctx.output / "celltype_counts.csv", counts)

    mapping_frame = pd.DataFrame(
        ctx.config["mapping"],
        columns=["original_r40_cluster", "recommended_label", "action", "reason"],
    )
    _save_table(ctx.output / "cortical_cluster_mapping.csv", mapping_frame)
    mapping_check = validate_cortical_mapping(ctx)
    ctx.report["checks"]["cortical_mapping"] = mapping_check
    cortical_keys = [key for key, _, _ in groups[:7]]
    ctx.report["checks"]["cortical_class_count"] = int(np.isin(labels, cortical_keys).sum())
    ctx.report["checks"]["cortical_class_names"] = cortical_keys
    ctx.report["outputs"].update({
        "celltype_counts": str(ctx.output / "celltype_counts.csv"),
        "cortical_cluster_mapping": str(ctx.output / "cortical_cluster_mapping.csv"),
    })

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        spatial = np.asarray(data["spatial"])
        umap = np.asarray(data.get("umap")) if data.get("umap") is not None else None
        for kind, coords, suffix in (
            ("spatial", spatial, "celltype_spatial.png"),
            ("umap", umap, "celltype_umap.png"),
        ):
            if coords is None:
                continue
            fig, ax = plt.subplots(figsize=(7.2, 4.4), facecolor="white")
            for key, _, color in groups:
                mask = labels == key
                if mask.any():
                    ax.scatter(coords[mask, 0], coords[mask, 1], s=0.15, c=color,
                               linewidths=0, rasterized=True)
            ax.set_title(f"Fig. 3 {kind} cells")
            ax.set_aspect("equal" if kind == "spatial" else "auto")
            if kind == "spatial":
                ax.set_axis_off()
            fig.tight_layout(pad=0.4)
            fig.savefig(ctx.output / suffix, dpi=300, facecolor="white")
            plt.close(fig)
            ctx.report["outputs"][suffix[:-4]] = str(ctx.output / suffix)
    except ImportError:
        ctx.report["unresolved"].append("matplotlib unavailable; celltype plots not rendered")

    ctx.report["stage"]["celltypes"] = "complete"

