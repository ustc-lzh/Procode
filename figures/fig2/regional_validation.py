"""Validate regional expression against the Allen atlas and render SFig. 3."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from data_io import (
    DISPLAY_REGIONS,
    REGIONS,
    _write_json,
    allen_region_dicts,
    load_allen_reference,
    load_h5ad_dataset,
    load_workbook_inputs,
    locate_h5ad_files,
)
from detection_metrics import cap_mean_metric, fpr_metric, wilcoxon_summary


REGION_PALETTE = {
    "OLF": "#43A047",
    "STR": "#D8432E",
    "Isocortex": "#F4A6A3",
    "CTXsp": "#B56576",
    "HPF": "#E58BC3",
    "HY": "#F2C078",
    "MB": "#A7D489",
    "HB": "#8D6E63",
    "TH": "#5E4FA2",
    "CB": "#2F78B7",
    "CB_IP": "#79B4D9",
    "fiber": "#BFD3EA",
    "VL": "#CFE8E8",
    "PAL": "#C58258",
    "TH_RT": "#3F2F7F",
    "TH_SPFp": "#9A89CA",
}

CONTROL_COLOR = "#159447"
OPTIMIZED_COLOR = "#CF1784"
MAIN_COLOR = "#D73027"
NEGATIVE_COLOR = "#18A6D1"
CONTEXT_COLOR = "#E5E5E5"


def load_h5ad_minimal(path: Path, is_bad: bool | None = None) -> dict[str, Any]:
    """Compatibility name for the archived loader, with current full fields."""

    if is_bad is None:
        is_bad = "bad" in Path(path).name.lower()
    return load_h5ad_dataset(Path(path), is_bad=is_bad)


def load_allen_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Compatibility name for the archived Allen loader.

    The current input is a workbook frame rather than an untracked CSV path.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Current regional validation expects the allen_reference workbook sheet")
    return load_allen_reference(frame)


def region_means(dataset: dict[str, Any], genes: list[str]) -> pd.DataFrame:
    indices = {str(gene): index for index, gene in enumerate(dataset["genes"])}
    missing = [gene for gene in genes if gene not in indices]
    if missing:
        raise ValueError(f"H5AD is missing genes required for regional correlation: {missing}")
    matrix = dataset["X"][:, [indices[gene] for gene in genes]]
    values = np.zeros((len(genes), len(REGIONS)), dtype=float)
    for index, region in enumerate(REGIONS):
        mask = dataset["analysis_regions"] == region
        if mask.any():
            values[:, index] = np.asarray(matrix[mask].mean(axis=0)).ravel()
    return pd.DataFrame(values, index=genes, columns=REGIONS)


def correlations(
    control: dict[str, Any], optimized: dict[str, Any], allen: pd.DataFrame
) -> pd.DataFrame:
    """Calculate the 91 gene-level Pearson pairs in the archived order."""

    common = sorted(
        set(map(str, allen.index))
        & set(map(str, control["genes"]))
        & set(map(str, optimized["genes"]))
    )
    if len(common) != 91:
        raise ValueError(f"Expected all 91 regional gene pairs; found {len(common)}")
    control_mean = region_means(control, common)
    optimized_mean = region_means(optimized, common)
    rows = []
    for gene in common:
        reference = allen.loc[gene, REGIONS].to_numpy(dtype=float)
        control_values = control_mean.loc[gene, REGIONS].to_numpy(dtype=float)
        optimized_values = optimized_mean.loc[gene, REGIONS].to_numpy(dtype=float)
        rows.append(
            {
                "gene": gene,
                "control_pearson_r": float(stats.pearsonr(control_values, reference).statistic),
                "sa_optimized_pearson_r": float(stats.pearsonr(optimized_values, reference).statistic),
            }
        )
    return pd.DataFrame(rows).sort_values("gene").reset_index(drop=True)


calculate_correlations = correlations


def _region_display_mask(correlations_frame: pd.DataFrame) -> tuple[np.ndarray, float, float]:
    values = correlations_frame[["control_pearson_r", "sa_optimized_pearson_r"]].to_numpy(dtype=float)
    lower, upper = np.percentile(values.ravel(), [5, 95])
    keep = ((values >= lower) & (values <= upper)).all(axis=1)
    return keep, float(lower), float(upper)


def p_label(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def render_regional_figure(
    control: dict[str, Any],
    optimized: dict[str, Any],
    allen: pd.DataFrame,
    correlations_frame: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    """Render standalone SFig. 3 maps and the correlation panel."""

    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:  # pragma: no cover - optional plotting dependency
        return {"status": "not_rendered", "reason": str(exc)}
    if control["spatial"] is None or optimized["spatial"] is None:
        return {"status": "not_rendered", "reason": "obsm/spatial is absent from an external H5AD"}

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    output_dir = Path(output_dir)
    fig = plt.figure(figsize=(180 / 25.4, 6.15), facecolor="white")

    def style_axis(ax: Any) -> None:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.invert_xaxis()

    def scale_bar(ax: Any, spatial: np.ndarray) -> None:
        x_range = float(np.ptp(spatial[:, 0]))
        fraction = (1000 / 0.1625) / x_range
        x2, y = 0.94, 0.065
        x1 = x2 - fraction
        ax.plot(
            [x1, x2],
            [y, y],
            transform=ax.transAxes,
            color="black",
            lw=1.4,
            solid_capstyle="butt",
            clip_on=False,
        )
        ax.text(
            (x1 + x2) / 2,
            y + 0.027,
            "1 mm",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=5.8,
        )

    def plot_map(ax: Any, dataset: dict[str, Any], title: str) -> None:
        for region in DISPLAY_REGIONS:
            mask = dataset["display_regions"] == region
            if mask.any():
                ax.scatter(
                    dataset["spatial"][mask, 0],
                    dataset["spatial"][mask, 1],
                    s=0.22,
                    c=REGION_PALETTE[region],
                    linewidths=0,
                    rasterized=True,
                )
        style_axis(ax)
        ax.set_title(title, fontsize=7.2, pad=1.5)
        scale_bar(ax, dataset["spatial"])

    ax_a1 = fig.add_axes([0.065, 0.705, 0.405, 0.245])
    ax_a2 = fig.add_axes([0.505, 0.705, 0.405, 0.245])
    plot_map(ax_a1, control, "Control codebook")
    plot_map(ax_a2, optimized, "SA-optimized codebook")
    fig.text(0.018, 0.972, "A", fontsize=10, fontweight="bold", va="top")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            label=region,
            markerfacecolor=REGION_PALETTE[region],
            markeredgecolor="none",
            markersize=4.2,
        )
        for region in DISPLAY_REGIONS
    ]
    fig.legend(
        handles=handles,
        title="Brain region",
        ncol=7,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.700),
        fontsize=5.9,
        title_fontsize=6.4,
        handletextpad=0.25,
        columnspacing=0.75,
        borderaxespad=0,
        labelspacing=0.38,
    )

    def plot_gene_regions(
        ax: Any, dataset: dict[str, Any], main_regions: list[str], negative_region: str
    ) -> None:
        labels = dataset["analysis_regions"]
        category = np.full(labels.shape, "context", dtype=object)
        category[labels == negative_region] = "negative"
        category[np.isin(labels, main_regions)] = "main"
        for name, color, size, alpha in (
            ("context", CONTEXT_COLOR, 0.18, 0.42),
            ("negative", NEGATIVE_COLOR, 0.24, 1.0),
            ("main", MAIN_COLOR, 0.24, 1.0),
        ):
            mask = category == name
            if mask.any():
                ax.scatter(
                    dataset["spatial"][mask, 0],
                    dataset["spatial"][mask, 1],
                    s=size,
                    c=color,
                    alpha=alpha,
                    linewidths=0,
                    rasterized=True,
                )
        style_axis(ax)
        scale_bar(ax, dataset["spatial"])

    b_left, b_width, b_gap = 0.105, 0.255, 0.020
    b_bottoms = [0.365, 0.120]
    for row, gene in enumerate(["Wfs1", "Drd1"]):
        if gene not in allen.index:
            continue
        top3 = allen.loc[gene].sort_values(ascending=False).head(3).index.tolist()
        lowest = str(allen.loc[gene].idxmin())
        for col, dataset in enumerate((control, optimized)):
            ax = fig.add_axes(
                [b_left + col * (b_width + b_gap), b_bottoms[row], b_width, 0.190]
            )
            plot_gene_regions(ax, dataset, top3, lowest)
        fig.text(
            0.058,
            b_bottoms[row] + 0.095,
            gene,
            fontsize=8.2,
            fontstyle="italic",
            va="center",
            ha="center",
        )
    fig.text(0.018, 0.620, "B", fontsize=10, fontweight="bold", va="top")
    fig.text(b_left + b_width / 2, 0.590, "Control codebook", fontsize=7.2, ha="center", va="bottom")
    fig.text(
        b_left + b_width + b_gap + b_width / 2,
        0.590,
        "SA-optimized codebook",
        fontsize=7.2,
        ha="center",
        va="bottom",
    )
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor=MAIN_COLOR,
                markeredgecolor="none",
                markersize=4.2,
                label="Main-expression regions (top 3)",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor=NEGATIVE_COLOR,
                markeredgecolor="none",
                markersize=4.2,
                label="Reference-negative region (lowest 1)",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor=CONTEXT_COLOR,
                markeredgecolor="#BDBDBD",
                markeredgewidth=0.3,
                markersize=4.2,
                label="Other regions (context)",
            ),
        ],
        ncol=3,
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.090, 0.045),
        fontsize=5.7,
        handletextpad=0.25,
        columnspacing=0.65,
        borderaxespad=0,
    )

    ax_c = fig.add_axes([0.715, 0.135, 0.255, 0.420])
    values = correlations_frame[["control_pearson_r", "sa_optimized_pearson_r"]].to_numpy(dtype=float)
    display_mask, low, high = _region_display_mask(correlations_frame)
    control_display, optimized_display = values[display_mask, 0], values[display_mask, 1]
    box = ax_c.boxplot(
        [control_display, optimized_display],
        positions=[1, 2],
        widths=0.48,
        whis=[5, 95],
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "lw": 1.1},
        whiskerprops={"color": "black", "lw": 0.8},
        capprops={"color": "black", "lw": 0.8},
        boxprops={"edgecolor": "black", "lw": 0.8},
    )
    for patch, color in zip(box["boxes"], [CONTROL_COLOR, OPTIMIZED_COLOR]):
        patch.set_facecolor(color)
        patch.set_alpha(0.90)
    rng = np.random.default_rng(42)
    jitter = rng.normal(0, 0.045, int(display_mask.sum()))
    ax_c.scatter(
        1 + jitter,
        control_display,
        s=6.0,
        facecolors=CONTROL_COLOR,
        edgecolors="black",
        linewidths=0.20,
        alpha=0.65,
        zorder=3,
    )
    ax_c.scatter(
        2 + jitter,
        optimized_display,
        s=6.0,
        facecolors=OPTIMIZED_COLOR,
        edgecolors="black",
        linewidths=0.20,
        alpha=0.65,
        zorder=3,
    )
    ax_c.set_xlim(0.48, 2.52)
    display_range = float(high - low)
    ax_c.set_ylim(low - 0.10 * display_range, min(1.17, high + 0.22 * display_range))
    ax_c.set_xticks([1, 2], ["Control", "SA-optimized"])
    ax_c.set_ylabel("Pearson correlation with Allen atlas", labelpad=2.5)
    ax_c.spines[["top", "right"]].set_visible(False)
    ax_c.tick_params(axis="both", labelsize=6.2, width=0.8, length=2.5)
    test = stats.wilcoxon(values[:, 0], values[:, 1], alternative="two-sided")
    y = high + 0.055 * display_range
    h = 0.040 * display_range
    ax_c.plot([1, 1, 2, 2], [y, y + h, y + h, y], color="black", lw=0.8, clip_on=False)
    ax_c.text(
        1.5,
        y + h + 0.025,
        p_label(float(test.pvalue)),
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        clip_on=False,
    )
    ax_c.text(
        1.5,
        low - 0.075 * display_range,
        f"{int(display_mask.sum())} of {len(correlations_frame)} gene pairs shown",
        ha="center",
        va="bottom",
        fontsize=5.5,
    )
    fig.text(0.680, 0.620, "C", fontsize=10, fontweight="bold", va="top")

    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in ((".pdf", {"dpi": 600}), (".svg", {"dpi": 600}), ("_600dpi.png", {"dpi": 600})):
        fig.savefig(output_dir / f"regional_validation{suffix}", facecolor="white", bbox_inches=None, **kwargs)
    plt.close(fig)
    return {
        "status": "rendered",
        "files": ["regional_validation.pdf", "regional_validation.svg", "regional_validation_600dpi.png"],
    }


def run_regional(
    inputs: dict[str, Any],
    data_root: Path | None,
    output_dir: Path,
    datasets: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run 91-pair regional correlations, FPR/CAP tables and SFig. 3."""

    bad_path, good_path = locate_h5ad_files(data_root)
    if datasets is None:
        datasets = {
            "bad": load_h5ad_dataset(bad_path, is_bad=True),
            "good": load_h5ad_dataset(good_path, is_bad=False),
        }
    control = datasets["bad"]
    optimized = datasets["good"]
    allen = load_allen_reference(inputs["allen_frame"])
    correlations_frame = correlations(control, optimized, allen)
    display_mask, display_lower, display_upper = _region_display_mask(correlations_frame)
    correlations_frame["included_in_statistical_test"] = True
    correlations_frame["included_in_display_p5_p95"] = display_mask
    corr_values = correlations_frame[["control_pearson_r", "sa_optimized_pearson_r"]].to_numpy(dtype=float)
    corr_test = stats.wilcoxon(corr_values[:, 0], corr_values[:, 1], alternative="two-sided")
    control_means = region_means(control, correlations_frame.gene.tolist())
    optimized_means = region_means(optimized, correlations_frame.gene.tolist())
    region_rows = []
    for source, values, scale in (
        ("Control", control_means, "mean log1p(raw counts) after QC"),
        ("SA-optimized", optimized_means, "mean log1p(raw counts) after QC"),
        ("Allen", allen, "Allen reference log2"),
    ):
        for gene in correlations_frame.gene:
            for region in REGIONS:
                region_rows.append(
                    {
                        "source": source,
                        "gene": gene,
                        "brain_region": region,
                        "expression": float(values.loc[gene, region]),
                        "scale": scale,
                    }
                )
    definitions_top, definitions_negative = allen_region_dicts(allen)
    definitions = pd.DataFrame(
        [
            {
                "gene": gene,
                "major_expression_regions": ";".join(definitions_top[gene]),
                "reference_negative_region": ";".join(definitions_negative[gene]),
            }
            for gene in correlations_frame.gene
        ]
    )
    out = Path(output_dir) / "sfig3"
    out.mkdir(parents=True, exist_ok=True)
    correlations_frame.to_csv(out / "sfig3_gene_level_pearson_correlations.csv", index=False)
    pd.DataFrame(region_rows).to_csv(out / "sfig3_region_expression_for_correlation.csv", index=False)
    definitions.to_csv(out / "gene_region_definitions.csv", index=False)

    common = correlations_frame.gene.tolist()
    fpr_bad = np.asarray([fpr_metric(control, gene, definitions_negative[gene]) for gene in common])
    fpr_good = np.asarray([fpr_metric(optimized, gene, definitions_negative[gene]) for gene in common])
    cap_bad = np.asarray([cap_mean_metric(control, gene, definitions_top[gene]) for gene in common])
    cap_good = np.asarray([cap_mean_metric(optimized, gene, definitions_top[gene]) for gene in common])
    fpr_summary = wilcoxon_summary(fpr_bad, fpr_good)
    cap_summary = wilcoxon_summary(cap_bad, cap_good)
    pd.DataFrame(
        {
            "gene": common,
            "control": fpr_bad,
            "sa_optimized": fpr_good,
            "included_in_statistical_test": True,
            "included_in_display_p5_p95": fpr_summary["plot_keep_mask"],
        }
    ).to_csv(out / "regional_fpr_paired_source_data.csv", index=False)
    pd.DataFrame(
        {
            "gene": common,
            "control": cap_bad,
            "sa_optimized": cap_good,
            "included_in_statistical_test": True,
            "included_in_display_p5_p95": cap_summary["plot_keep_mask"],
        }
    ).to_csv(out / "regional_cap_mean_paired_source_data.csv", index=False)

    regional_stats = {
        "method": "Pearson r over 11 matched regions per gene; paired Wilcoxon across all genes",
        "alternative": "two-sided",
        "n_test": int(len(correlations_frame)),
        "n_display": int(display_mask.sum()),
        "statistic": float(corr_test.statistic),
        "p_computed": float(corr_test.pvalue),
        "display_p05": display_lower,
        "display_p95": display_upper,
        "regions": REGIONS,
        "qc_cells": {"control": control["n_after_qc"], "sa_optimized": optimized["n_after_qc"]},
        "plot_filter_applies_to_display_only": True,
        "fpr": fpr_summary,
        "cap_mean": cap_summary,
    }
    _write_json(out / "sfig3_correlation_statistics.json", regional_stats)
    render_status = render_regional_figure(control, optimized, allen, correlations_frame, out)
    regional_stats["figure"] = render_status
    _write_json(out / "sfig3_correlation_statistics.json", regional_stats)
    return regional_stats


# Archived name used by the 20261019 module; the current stage writes the
# standalone files through render_regional_figure and keeps this alias narrow.
build_figure = render_regional_figure


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=Path(__file__).resolve().parent / "fig2_data.xlsx")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args(argv)
    try:
        inputs = load_workbook_inputs(args.workbook)
        run_regional(inputs, args.data_root, args.output_dir)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"regional_validation.py: error: {exc}") from exc


if __name__ == "__main__":
    main()
