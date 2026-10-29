"""Nuclear label expansion and assignment of transcript coordinates to cells."""

from __future__ import annotations

import numpy as np


def scale_to_uint8(image: np.ndarray) -> np.ndarray:
    """Scale an image to uint8 for Mesmer input."""

    values = np.asarray(image)
    if values.ndim == 0:
        raise ValueError("image must be an array")
    maximum = float(np.max(values))
    if maximum <= 0:
        return np.zeros(values.shape, dtype=np.uint8)
    normalized = values.astype(float) / maximum
    return np.clip(normalized * 255, 0, 255).astype(np.uint8)


def expand_labels_no_merge(
    labels: np.ndarray,
    distance: int,
    max_area: int,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Expand labels by 4-neighbour steps without merging different labels.

    An unlabeled pixel is filled only when its current 4-neighbourhood contains
    exactly one positive label. The area limit is checked before each assignment.
    """

    values = np.asarray(labels)
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("labels must be a two-dimensional integer array")
    if distance < 0 or max_area < 1:
        raise ValueError("distance must be non-negative and max_area positive")
    allowed = np.ones(values.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if allowed.shape != values.shape:
        raise ValueError("mask must have the same shape as labels")
    expanded = values.copy()
    areas = np.bincount(expanded.ravel(), minlength=int(expanded.max()) + 1)
    height, width = expanded.shape
    for _ in range(distance):
        previous = expanded.copy()
        additions: dict[int, int] = {}
        for row in range(height):
            for column in range(width):
                if previous[row, column] != 0 or not allowed[row, column]:
                    continue
                neighbors: list[int] = []
                if row:
                    neighbors.append(int(previous[row - 1, column]))
                if row + 1 < height:
                    neighbors.append(int(previous[row + 1, column]))
                if column:
                    neighbors.append(int(previous[row, column - 1]))
                if column + 1 < width:
                    neighbors.append(int(previous[row, column + 1]))
                candidate = sorted({label for label in neighbors if label > 0})
                if len(candidate) == 1 and areas[candidate[0]] < max_area:
                    expanded[row, column] = candidate[0]
                    additions[candidate[0]] = additions.get(candidate[0], 0) + 1
        for label, count in additions.items():
            areas[label] += count
    return expanded


def map_points_to_labels(
    points: np.ndarray, labels: np.ndarray, *, on_outside: int = 0
) -> np.ndarray:
    """Map integer ``(y, x)`` points to a 2-D label image, safely."""

    coordinates = np.asarray(points)
    image = np.asarray(labels)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("points must have shape (n_points, 2) in y,x order")
    if image.ndim != 2:
        raise ValueError("labels must be two-dimensional")
    output = np.full(coordinates.shape[0], on_outside, dtype=image.dtype)
    for index, (row, column) in enumerate(coordinates):
        row_i, column_i = int(row), int(column)
        if 0 <= row_i < image.shape[0] and 0 <= column_i < image.shape[1]:
            output[index] = image[row_i, column_i]
    return output
