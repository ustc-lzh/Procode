"""Background correction and intensity normalization for image arrays."""

from __future__ import annotations

import numpy as np


def percentile_background(images: np.ndarray, percentile: float = 75.0) -> np.ndarray:
    """Return a pixel-wise percentile background from ``(n, y, x)`` images."""

    values = np.asarray(images)
    if values.ndim != 3 or values.shape[0] < 1:
        raise ValueError("images must have shape (n_images, height, width)")
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100")
    return np.percentile(values, percentile, axis=0)


def mean_scaled_flat_field(
    image: np.ndarray, background: np.ndarray, eps: float = 1e-12
) -> np.ndarray:
    """Divide an image by a supplied background and restore its mean intensity.

    Apply any required smoothing to the background before calling this function.
    """

    image_array = np.asarray(image, dtype=float)
    background_array = np.asarray(background, dtype=float)
    if image_array.shape != background_array.shape:
        raise ValueError("image and background must have the same shape")
    if image_array.ndim != 2:
        raise ValueError("image and background must be two-dimensional")
    denominator = np.maximum(background_array, eps)
    return (image_array / denominator) * float(np.mean(background_array))


def normalize_image_sequence(images: np.ndarray, percentile: float = 90.0) -> np.ndarray:
    """Clip a sequence and mean-scale each image, matching the source helper."""

    values = np.asarray(images, dtype=float)
    if values.ndim != 3 or values.shape[0] < 1:
        raise ValueError("images must have shape (n_images, height, width)")
    clip_max = float(np.percentile(values, percentile))
    clipped = np.clip(values, 0, clip_max)
    means = np.mean(clipped, axis=(1, 2))
    global_mean = float(np.mean(means))
    factors = np.divide(
        global_mean,
        means,
        out=np.ones_like(means, dtype=float),
        where=means != 0,
    )
    return clipped * factors[:, None, None]


def crop_edges(image: np.ndarray, edge: int = 100) -> np.ndarray:
    """Crop ``edge`` pixels on all sides, before spot detection."""

    values = np.asarray(image)
    if values.ndim < 2:
        raise ValueError("image must have at least two dimensions")
    if edge < 0 or 2 * edge >= min(values.shape[-2:]):
        raise ValueError("edge would produce an empty image")
    return values[..., edge:-edge, edge:-edge] if edge else values.copy()
