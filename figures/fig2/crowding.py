"""Calculate abundance-weighted optical crowding for both codebooks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from data_io import _write_json, load_workbook_inputs


def evaluate(
    expression: pd.DataFrame,
    abundance: pd.Series,
    books: dict[str, pd.DataFrame],
    common: pd.Index,
    cutoff: float,
    top_n: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate the original abundance-weighted bit-crowding definition.

    ``common`` is explicit because the workbook validates that the target
    expression and both codebooks contain the same 96 unique genes before
    this function is called.
    """

    summary: list[dict[str, Any]] = []
    detail: list[dict[str, Any]] = []
    selection: list[dict[str, Any]] = []
    expression = expression.loc[common]
    for cell_type in expression.columns:
        selected = expression.loc[:, cell_type]
        selected = selected[selected > cutoff].sort_values(ascending=False, kind="stable")
        if top_n is not None:
            selected = selected.iloc[:top_n]
        selection.extend(
            {
                "CellType": cell_type,
                "Gene": gene,
                "Expression": float(value),
                "Threshold": float(cutoff),
            }
            for gene, value in selected.items()
        )
        row: dict[str, Any] = {
            "CellType": cell_type,
            "Abundance": float(abundance[cell_type]),
            "SelectedGeneCount": int(len(selected)),
            "SelectedGenes": ",".join(selected.index),
            "Threshold": float(cutoff),
        }
        for name, codebook in books.items():
            total = 0.0
            group_score = 0.0
            crowded_bit_count = 0
            for bit in codebook.columns:
                members = selected.index[codebook.loc[selected.index, bit].to_numpy() > 0]
                count = len(members)
                if count < 2:
                    continue
                crowded_bit_count += 1
                load = float(selected.loc[members].sum())
                total += load
                group_score += load * (count - 1)
                detail.append(
                    {
                        "CellType": cell_type,
                        "Codebook": name,
                        "Bit": bit,
                        "GeneCountInBit": int(count),
                        "Genes": ",".join(members),
                        "ExprValues": json.dumps(selected.loc[members].to_dict(), ensure_ascii=False),
                        "TotalExprLoad": load,
                        "GroupScore": load * (count - 1),
                        "Abundance": float(abundance[cell_type]),
                        "GroupScore_AbundanceWeight": load * float(abundance[cell_type]),
                    }
                )
            row[name] = total * float(abundance[cell_type])
            row[f"{name}_CrowdedBitCount"] = int(crowded_bit_count)
            row[f"{name}_GroupScore"] = group_score
        row["Delta_Control_minus_SA"] = row["Control"] - row["SA-optimized"]
        summary.append(row)
    return pd.DataFrame(summary), pd.DataFrame(detail), pd.DataFrame(selection)


def summarize(scores: pd.DataFrame) -> dict[str, Any]:
    """Summarise the paired 34-cell-type crowding comparison."""

    values = scores[["Control", "SA-optimized"]].to_numpy(dtype=float)
    delta = values[:, 0] - values[:, 1]
    if np.all(delta == 0):
        statistic, p_value = 0.0, 1.0
    else:
        result = wilcoxon(
            values[:, 0],
            values[:, 1],
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        )
        statistic, p_value = float(result.statistic), float(result.pvalue)
    lower, upper = np.percentile(values.ravel(), [5, 95], method="linear")
    keep = ((values >= lower) & (values <= upper)).all(axis=1)
    return {
        "test": "two-sided paired Wilcoxon signed-rank",
        "zero_method": "wilcox",
        "method": "auto",
        "statistic": statistic,
        "p": p_value,
        "n_test": int(len(scores)),
        "n_nonzero_differences": int(np.count_nonzero(delta)),
        "n_plot": int(keep.sum()),
        "plot_mask": keep.tolist(),
        "display_p05": float(lower),
        "display_p95": float(upper),
        "n_control_greater": int((delta > 0).sum()),
        "n_SA_greater": int((delta < 0).sum()),
        "control_mean": float(values[:, 0].mean()),
        "SA_mean": float(values[:, 1].mean()),
        "control_median": float(np.median(values[:, 0])),
        "SA_median": float(np.median(values[:, 1])),
        "sum_control": float(values[:, 0].sum()),
        "sum_SA": float(values[:, 1].sum()),
        "celltypes": scores.CellType.tolist(),
        "bad": values[:, 0].tolist(),
        "good": values[:, 1].tolist(),
    }


def verify_matrix(
    expression: pd.DataFrame,
    abundance: pd.Series,
    books: dict[str, pd.DataFrame],
    common: pd.Index,
    cutoff: float,
    scores: pd.DataFrame,
) -> None:
    """Independently verify the abundance-weighted score in matrix form."""

    expression_values = expression.loc[common].to_numpy(dtype=float)
    selected = expression_values > cutoff
    for name, codebook in books.items():
        bit_matrix = codebook.loc[common].to_numpy(dtype=int)
        counts = selected.T.astype(int) @ bit_matrix
        loads = np.where(selected, expression_values, 0.0).T @ bit_matrix
        expected = (loads * (counts >= 2)).sum(axis=1) * abundance.to_numpy(dtype=float)
        np.testing.assert_allclose(
            expected,
            scores[name].to_numpy(dtype=float),
            rtol=1e-12,
            atol=1e-12,
        )


def run_crowding(inputs: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Run the crowding stage from the validated compact workbook inputs."""

    crowd_dir = Path(output_dir) / "crowding"
    crowd_dir.mkdir(parents=True, exist_ok=True)
    background = inputs["background"].copy()
    target_expression = inputs["target_expression"]
    books = inputs["books"]
    abundance = inputs["abundance"]
    common = target_expression.index

    # The threshold uses every original background row, including duplicate
    # non-target symbols.  It is intentionally not recomputed from 96 targets.
    cutoff = float(pd.Series(background["PeakExpression"]).quantile(0.95, interpolation="linear"))
    background["AboveGlobalP95"] = background["PeakExpression"].to_numpy(dtype=float) > cutoff
    background["InBothCodebooks"] = background["Gene"].isin(common)
    background.to_csv(crowd_dir / "gene_peak_expression.csv", index=False, encoding="utf-8-sig")

    scores, detail, selection = evaluate(target_expression, abundance, books, common, cutoff)
    verify_matrix(target_expression, abundance, books, common, cutoff, scores)
    comparison = summarize(scores)
    scores["IncludedInPlot_5_95"] = comparison["plot_mask"]
    scores.to_csv(crowd_dir / "crowding_scores.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(crowd_dir / "crowded_bits.csv", index=False, encoding="utf-8-sig")
    selection.to_csv(crowd_dir / "selected_genes.csv", index=False, encoding="utf-8-sig")

    statistics = {
        "definition": "T = Q_0.95({max_t E[g,t] for every original reference row g}); select E[g,t] > T in each cell type; no top-N cap",
        "threshold": cutoff,
        "threshold_scope": "all",
        "threshold_scope_description": "All 30,934 original reference rows, retaining duplicate non-target symbols",
        "n_threshold_genes": int(len(background)),
        "n_expression_genes": int(len(background)),
        "n_common_codebook_genes": int(len(common)),
        "n_celltypes": int(target_expression.shape[1]),
        "n_unique_expression_gene_names": int(background["Gene"].nunique()),
        "duplicated_non_target_gene_names": sorted(
            set(background.loc[background["Gene"].duplicated(keep=False), "Gene"]) - set(common)
        ),
        "duplicate_policy": "Keep all original reference rows for threshold calculation; target genes are unique and are not aggregated.",
        "all_expression_peak_p95": cutoff,
        "common_codebook_peak_p95": float(target_expression.max(axis=1).quantile(0.95, interpolation="linear")),
        "codebook_missing_expression": {
            name: sorted(set(book.index) - set(target_expression.index)) for name, book in books.items()
        },
        "codebook_gene_counts": {name: int(len(book)) for name, book in books.items()},
        "sum_abundance": float(abundance.sum()),
        "score": "abundance[t] * sum_over_bits(I[k>=2] * sum_selected_gene_expression_in_bit)",
        "workbook": inputs["workbook"],
        "workbook_sheets": inputs["sheets"],
        "comparison": comparison,
        "independent_matrix_check": "passed",
        "n_celltypes_with_selected_genes": int((scores.SelectedGeneCount > 0).sum()),
        "max_selected_gene_count_per_celltype": int(scores.SelectedGeneCount.max()),
    }
    _write_json(crowd_dir / "statistics.json", statistics)
    return statistics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=Path(__file__).resolve().parent / "fig2_data.xlsx")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args(argv)
    try:
        run_crowding(load_workbook_inputs(args.workbook), args.output_dir)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"crowding.py: error: {exc}") from exc


if __name__ == "__main__":
    main()
