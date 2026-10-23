"""Binary codeword decoding and spatial merging of detected spots."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def encode_from_read(df_read: pd.DataFrame) -> pd.DataFrame:
    """Convert a round -> gene table to a binary gene x round table."""

    all_genes: set[Any] = set()
    for column in df_read.columns:
        for value in df_read[column].dropna():
            if value != 0:
                all_genes.add(value)
    genes = sorted(all_genes)
    result = pd.DataFrame(0, index=genes, columns=df_read.columns, dtype=int)
    for column in df_read.columns:
        for gene in df_read[column].dropna():
            if gene != 0:
                result.at[gene, column] = 1
    return result


def ball_tree_decode(
    codebook_all: pd.DataFrame,
    result_matrix: pd.DataFrame,
    distance_cut: float = 0.7071,
    top_number: int = 2,
) -> pd.DataFrame:
    """Return nearest codewords within the source-specified Euclidean radius.

    The source notebooks use ``sklearn.neighbors.BallTree``. The import is lazy so
    this module remains importable in a minimal static-check environment.
    """

    if distance_cut < 0 or top_number < 1:
        raise ValueError("distance_cut must be non-negative and top_number positive")
    if list(codebook_all.columns) != list(result_matrix.columns):
        raise ValueError("codebook and result matrix columns must match")
    if codebook_all.empty or result_matrix.empty:
        return pd.DataFrame(columns=["spotid", "gene", "distance"])
    codebook_values = codebook_all.to_numpy(dtype=float)
    try:
        from sklearn.neighbors import BallTree
    except ModuleNotFoundError:  # pragma: no cover - exercised in minimal runtimes
        BallTree = None  # type: ignore[assignment]

    tree = BallTree(codebook_values, metric="euclidean") if BallTree else None
    rows: list[dict[str, Any]] = []
    for spot_id, row in result_matrix.iterrows():
        vector = row.to_numpy(dtype=float)
        if tree is not None:
            indices, distances = tree.query_radius(
                vector[None, :],
                r=distance_cut,
                return_distance=True,
                sort_results=True,
            )
            nearest = zip(indices[0][:top_number], distances[0][:top_number])
        else:
            distances_array = np.linalg.norm(codebook_values - vector, axis=1)
            candidate_indices = np.flatnonzero(distances_array <= distance_cut)
            candidate_indices = candidate_indices[
                np.argsort(distances_array[candidate_indices])
            ][:top_number]
            nearest = ((index, distances_array[index]) for index in candidate_indices)
        for index, distance in nearest:
            if distance <= distance_cut:
                rows.append(
                    {
                        "spotid": spot_id,
                        "gene": codebook_all.index[index],
                        "distance": float(distance),
                    }
                )
    return pd.DataFrame(rows, columns=["spotid", "gene", "distance"])


def merge_points_within_distance(
    points: pd.DataFrame, max_distance: float = 3.0
) -> pd.DataFrame:
    """Keep the first point in each FOV/coordinate DBSCAN cluster.

    This mirrors the small helper in the 304-gene decode notebook. It expects
    ``fov``, ``x`` and ``y`` columns and returns a copy with a ``dup`` count.
    """

    required = {"fov", "x", "y"}
    missing = required - set(points.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if max_distance < 0:
        raise ValueError("max_distance must be non-negative")
    output: list[pd.DataFrame] = []
    for fov, group in points.groupby("fov", sort=False):
        current = group.copy()
        coordinates = current[["x", "y"]].to_numpy(dtype=float)
        try:
            from sklearn.cluster import DBSCAN
        except ModuleNotFoundError:  # pragma: no cover - minimal runtimes
            # A tiny connected-components fallback equivalent to DBSCAN with
            # min_samples=1. This keeps static helpers usable without sklearn.
            cluster_ids = np.full(len(coordinates), -1, dtype=int)
            next_cluster = 0
            for start in range(len(coordinates)):
                if cluster_ids[start] >= 0:
                    continue
                cluster_ids[start] = next_cluster
                pending = [start]
                while pending:
                    source = pending.pop()
                    distances = np.linalg.norm(coordinates - coordinates[source], axis=1)
                    neighbors = np.flatnonzero(distances <= max_distance)
                    for neighbor in neighbors:
                        if cluster_ids[neighbor] < 0:
                            cluster_ids[neighbor] = next_cluster
                            pending.append(int(neighbor))
                next_cluster += 1
            current["cluster"] = cluster_ids
        else:
            current["cluster"] = DBSCAN(
                eps=max_distance, min_samples=1, metric="euclidean"
            ).fit_predict(coordinates)
        counts = current["cluster"].value_counts()
        current["dup"] = current["cluster"].map(counts)
        output.append(current.groupby("cluster", sort=False).first().reset_index(drop=True))
    if not output:
        return points.iloc[0:0].copy()
    return pd.concat(output, ignore_index=True)
