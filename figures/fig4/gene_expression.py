#!/usr/bin/env python3
"""Count-matrix preparation and the release's 34-factor normalization.

This module deliberately keeps matrix work separate from marker plotting.  It
validates the retained-cell join before converting raw counts to the exact
normalization used by the packaged marker table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_io import find_column, normalize_ids


NORMALIZATION_FACTOR = 34.0


def counts_columns(counts: pd.DataFrame) -> tuple[str, list[str]]:
    """Return the cell-ID column and gene columns from a raw count table."""

    id_column = find_column(counts, "cell", "cell_id", "cellid")
    if id_column is None:
        id_column = str(counts.columns[0])
    genes = [str(column) for column in counts.columns if str(column) != id_column]
    return id_column, genes


def retained_counts(
    counts: pd.DataFrame,
    metadata: pd.DataFrame,
    genes: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join raw counts to retained metadata and return counts plus metadata.

    The order is the canonical retained metadata order, which makes marker
    statistics reproducible and avoids accidental positional joins.
    """

    id_column, available_genes = counts_columns(counts)
    missing_genes = [gene for gene in genes if gene not in available_genes]
    if missing_genes:
        raise ValueError(f"Raw counts are missing dotplot genes: {missing_genes}")
    table = counts.copy()
    table[id_column] = normalize_ids(table[id_column])
    if table[id_column].duplicated().any():
        raise ValueError("Raw counts contain duplicate cell identifiers.")
    table = table.set_index(id_column)
    retained = metadata.loc[metadata["retained_for_final_figure"]].copy()
    retained_ids = normalize_ids(retained["cell_id"])
    missing_ids = retained_ids.difference(table.index)
    if len(missing_ids):
        raise ValueError(f"Raw counts do not cover {len(missing_ids)} retained cells.")
    selected = table.loc[retained_ids, genes].apply(pd.to_numeric, errors="raise")
    retained = retained.set_index("cell_id").loc[retained_ids].copy()
    return selected, retained


def log_normalize_34(counts: pd.DataFrame, transcripts_per_cell: pd.Series) -> pd.DataFrame:
    """Apply ``log1p(raw / max(transcripts_per_cell, 1) * 34)``."""

    library = pd.to_numeric(transcripts_per_cell, errors="coerce")
    if library.isna().any():
        raise ValueError("transcripts_per_cell is required to recompute marker statistics.")
    values = counts.to_numpy(dtype=float)
    normalized = np.log1p(
        values / np.maximum(library.to_numpy(dtype=float), 1.0)[:, None] * NORMALIZATION_FACTOR
    )
    return pd.DataFrame(normalized, index=counts.index, columns=counts.columns)


def qc_values(metadata: pd.DataFrame) -> dict[str, pd.Series]:
    """Return QC vectors used by the independent Supplementary Fig. 6 plots."""

    genes = pd.to_numeric(metadata.get("genes_per_cell", pd.Series(dtype=float)), errors="coerce")
    transcripts = pd.to_numeric(
        metadata.get("transcripts_per_cell", pd.Series(dtype=float)), errors="coerce"
    )
    return {
        "genes_per_cell": genes,
        "transcripts_per_cell": transcripts,
        "log2_genes_per_cell_plus1": metadata.get(
            "log2_genes_per_cell_plus1", np.log2(genes + 1)
        ),
        "log2_transcripts_per_cell_plus1": metadata.get(
            "log2_transcripts_per_cell_plus1", np.log2(transcripts + 1)
        ),
    }

