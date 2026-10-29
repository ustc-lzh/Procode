"""Calculate per-cell detection and regional signal metrics for Figure 2."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from data_io import (
    _write_json,
    allen_region_dicts,
    load_allen_reference,
    load_h5ad_dataset,
    load_workbook_inputs,
    locate_h5ad_files,
)


def region_sum_count(
    dataset: dict[str, Any], gene_index: int, region_key: str
) -> tuple[dict[str, float], dict[str, int]]:
    values = np.asarray(dataset["X"][:, gene_index].toarray()).ravel().astype(float)
    values = np.nan_to_num(values, nan=0.0)
    values = np.clip(values, a_min=0.0, a_max=None)
    labels = dataset[region_key]
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for label, value in zip(labels, values):
        key = str(label)
        sums[key] = sums.get(key, 0.0) + float(value)
        counts[key] = counts.get(key, 0) + 1
    return sums, counts


def _gene_index(dataset: dict[str, Any]) -> dict[str, int]:
    return {str(gene): index for index, gene in enumerate(dataset["genes"])}


def fpr_metric(dataset: dict[str, Any], gene: str, wrong_regions: list[str]) -> float:
    index = _gene_index(dataset)[gene]
    sums, _ = region_sum_count(dataset, index, "region2")
    total = float(sum(sums.values()))
    false_sum = float(sum(sums.get(str(region), 0.0) for region in wrong_regions))
    return false_sum / total if total > 0 else 0.0


def cap_mean_metric(dataset: dict[str, Any], gene: str, top_regions: list[str]) -> float:
    index = _gene_index(dataset)[gene]
    sums, counts = region_sum_count(dataset, index, "region3")
    top_sum = float(sum(sums.get(str(region), 0.0) for region in top_regions))
    top_cells = int(sum(counts.get(str(region), 0) for region in top_regions))
    return top_sum / top_cells if top_cells > 0 else 0.0


def wilcoxon_summary(bad: np.ndarray, good: np.ndarray) -> dict[str, Any]:
    bad = np.asarray(bad, dtype=float)
    good = np.asarray(good, dtype=float)
    differences = good - bad
    if np.all(differences == 0):
        statistic, p_value = 0.0, 1.0
    else:
        result = stats.wilcoxon(bad, good, alternative="two-sided")
        statistic, p_value = float(result.statistic), float(result.pvalue)
    combined = np.concatenate([bad, good])
    lower, upper = np.percentile(combined, [5, 95])
    mask = (bad >= lower) & (bad <= upper) & (good >= lower) & (good <= upper)
    return {
        "statistic": statistic,
        "p_value": p_value,
        "alternative": "two-sided",
        "n_test": int(len(bad)),
        "n_plot": int(mask.sum()),
        "percentile_range": [5, 95],
        "plot_lower": float(lower),
        "plot_upper": float(upper),
        "n_excluded_plot": int((~mask).sum()),
        "differences_good_minus_bad": differences.tolist(),
        "plot_keep_mask": mask.tolist(),
    }


def summary_stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std_sample": float(np.std(values, ddof=1)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "q05": float(np.percentile(values, 5)),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
        "q95": float(np.percentile(values, 95)),
    }


def per_cell_statistics(good: dict[str, Any], bad: dict[str, Any], metric: str) -> dict[str, Any]:
    good_values = np.asarray(good[metric], dtype=float)
    bad_values = np.asarray(bad[metric], dtype=float)
    test = stats.mannwhitneyu(bad_values, good_values, alternative="two-sided")
    return {
        "metric": metric,
        "test": "Mann-Whitney U",
        "alternative": "two-sided",
        "statistic": float(test.statistic),
        "p_value": float(test.pvalue),
        "n_combined": int(good_values.size + bad_values.size),
        "bad": summary_stats(bad_values),
        "good": summary_stats(good_values),
        "plot_y_limits": [0, 88] if metric == "n_genes_by_counts" else [0, 290],
    }


def run_detection(
    inputs: dict[str, Any],
    data_root: Path | None,
    output_dir: Path,
    datasets: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Run QC, per-cell detection tests, FPR and CAP metrics."""

    bad_path, good_path = locate_h5ad_files(data_root)
    if datasets is None:
        datasets = {
            "bad": load_h5ad_dataset(bad_path, is_bad=True),
            "good": load_h5ad_dataset(good_path, is_bad=False),
        }
    bad = datasets["bad"]
    good = datasets["good"]
    allen = load_allen_reference(inputs["allen_frame"])
    top3, wrong = allen_region_dicts(allen)
    common_genes = sorted(
        set(map(str, good["genes"]))
        & set(map(str, bad["genes"]))
        & set(map(str, allen.index))
    )
    if len(common_genes) != 91:
        raise ValueError(f"Expected all 91 Allen/H5AD gene pairs; found {len(common_genes)}")

    fpr_bad = [fpr_metric(bad, gene, wrong[gene]) for gene in common_genes]
    fpr_good = [fpr_metric(good, gene, wrong[gene]) for gene in common_genes]
    cap_bad = [cap_mean_metric(bad, gene, top3[gene]) for gene in common_genes]
    cap_good = [cap_mean_metric(good, gene, top3[gene]) for gene in common_genes]
    fpr = wilcoxon_summary(np.asarray(fpr_bad), np.asarray(fpr_good))
    cap_mean = wilcoxon_summary(np.asarray(cap_bad), np.asarray(cap_good))
    fpr.update(
        {
            "metric": "fpr",
            "bad": fpr_bad,
            "good": fpr_good,
            "genes": common_genes,
            "region_definition": "brain_region2",
            "wrong_region_rule": "one Allen Log2 region with minimum expression per gene",
        }
    )
    cap_mean.update(
        {
            "metric": "cap_mean",
            "bad": cap_bad,
            "good": cap_good,
            "genes": common_genes,
            "region_definition": "brain_region3",
            "top_region_rule": "three Allen Log2 regions with highest expression per gene",
        }
    )

    output = {
        "provenance": {
            "script": "figures/fig2/analysis.py",
            "source_h5ad": {"good": str(good_path), "bad": str(bad_path)},
            "source_workbook": inputs["workbook"],
            "workbook_sheets": inputs["sheets"],
            "qc_rule": "drop var_names == blank; total_counts > 4 and < 250; n_genes_by_counts > 3; pixel_area > 1000 and < 20000",
            "computed_expression_scale": "log1p(raw counts) after QC",
        },
        "datasets": {
            name: {
                "path": dataset["path"],
                "n_before_qc": dataset["n_before_qc"],
                "n_after_qc": dataset["n_after_qc"],
                "n_dropped_qc": dataset["n_dropped_qc"],
                "n_blank_vars": dataset["n_blank_vars"],
                "n_vars_after_blank": dataset["n_vars_after_blank"],
            }
            for name, dataset in (("good", good), ("bad", bad))
        },
        "D": {
            "n_genes_by_counts": per_cell_statistics(good, bad, "n_genes_by_counts"),
            "total_counts": per_cell_statistics(good, bad, "total_counts"),
        },
        "E": {"fpr": fpr, "cap_mean": cap_mean},
        "B": {"status": "crowding stage is independent; see crowding/statistics.json"},
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cell_rows = []
    for name, dataset, condition in (("bad", bad, "Control"), ("good", good, "SA-optimized")):
        cell_rows.append(
            pd.DataFrame(
                {
                    "sample_id": f"R2_{name}",
                    "Codebook": condition,
                    "source_row_0based": dataset["source_rows"],
                    "cell_id": dataset["cell_ids"],
                    "DetectedGenes": dataset["n_genes_by_counts"],
                    "TotalCounts": dataset["total_counts"],
                    "pixel_area": dataset["pixel_area"],
                }
            )
        )
    pd.concat(cell_rows, ignore_index=True).to_csv(out / "per_cell_counts.csv", index=False)
    _write_json(out / "detection_metrics.json", output)
    return output, datasets


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=Path(__file__).resolve().parent / "fig2_data.xlsx")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args(argv)
    try:
        inputs = load_workbook_inputs(args.workbook)
        run_detection(inputs, args.data_root, args.output_dir)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"detection_metrics.py: error: {exc}") from exc


if __name__ == "__main__":
    main()
