"""Standalone QC and marker-expression plots for Fig. 3 supplements."""

from __future__ import annotations

from pathlib import Path


def render_qc(ctx, data) -> None:
    """Summarize cell-quality metrics and plot their distributions."""

    np, pd, _, _, _ = _deps()
    qc_rows = []
    for column, label in (("n_genes_by_counts", "genes_per_cell"),
                          ("total_counts", "transcripts_per_cell")):
        values = data.get(column)
        if values is None:
            continue
        values = np.asarray(values, dtype=float)
        qc_rows.append((label, int(len(values)), float(np.mean(values)), float(np.median(values)),
                        float(np.min(values)), float(np.max(values))))
    if not qc_rows:
        return
    qc_frame = pd.DataFrame(qc_rows, columns=["metric", "n_cells", "mean", "median", "min", "max"])
    qc_path = ctx.output / "qc_summary.csv"
    qc_frame.to_csv(qc_path, index=False)
    ctx.report["formulas"]["qc_histogram"] = "log2(value + 1), 36 bins"
    ctx.report["outputs"]["qc_summary"] = str(qc_path)
    ctx.report["checks"]["qc_summary_rows"] = int(len(qc_frame))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(qc_rows), figsize=(7.2, 2.8), facecolor="white", squeeze=False)
        for ax, (column, label) in zip(axes.flat, (("n_genes_by_counts", "Genes per cell"),
                                                   ("total_counts", "Transcripts per cell"))):
            values = data.get(column)
            if values is None:
                ax.axis("off")
                continue
            ax.hist(np.log2(np.asarray(values, dtype=float) + 1), bins=36,
                    color="#5794C4", edgecolor="white", linewidth=.2)
            ax.set_title(label)
            ax.set_xlabel("log2(value + 1)")
            ax.set_ylabel("Cells")
        fig.tight_layout()
        path = ctx.output / "qc_histograms.png"
        fig.savefig(path, dpi=300, facecolor="white")
        plt.close(fig)
        ctx.report["outputs"]["qc_histograms"] = str(path)
    except ImportError:
        ctx.report["unresolved"].append("matplotlib unavailable; QC histograms not rendered")


def render_marker_dotplot(ctx, genes, keys, fractions, scaled) -> None:
    """Render the numeric marker statistics as a standalone dot plot."""

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        ctx.report["unresolved"].append("matplotlib unavailable; marker dot plot not rendered")
        return
    fractions = _array(fractions)
    scaled = _array(scaled)
    cmap = plt.get_cmap("Reds")
    fig, ax = plt.subplots(figsize=(max(10.0, .25 * len(genes)), max(4.5, .25 * len(keys))),
                           facecolor="white")
    for i in range(len(keys)):
        for j in range(len(genes)):
            size = (4.0 + 42.0 * fractions[i, j] ** 1.05) ** 2 / 4.0
            ax.scatter(j, i, s=size, c=[cmap(float(scaled[i, j]))],
                       edgecolors="#777777", linewidths=.15)
    ax.set_xticks(range(len(genes)), genes, rotation=68, ha="right")
    ax.set_yticks(range(len(keys)), keys)
    ax.set_xlim(-1, len(genes))
    ax.set_ylim(len(keys) - .5, -.5)
    ax.set_title("Marker expression in displayed populations")
    ax.set_xlabel("Marker gene")
    ax.set_ylabel("Display group")
    fig.tight_layout()
    path = ctx.output / "marker_dotplot.png"
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    ctx.report["outputs"]["marker_dotplot"] = str(path)


def _array(value):
    import numpy as np
    return np.asarray(value, dtype=float)


def _deps():
    try:
        import numpy as np  # noqa: F401
        import pandas as pd  # noqa: F401
        from .data_io import _deps as shared_deps
    except ImportError:
        from data_io import _deps as shared_deps
    return shared_deps()[0], shared_deps()[1], None, None, None

