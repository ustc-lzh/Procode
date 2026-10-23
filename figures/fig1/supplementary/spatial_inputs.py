"""Read cell-centroid spatial inputs from H5AD, with optional NPZ fallback.

The released code package does not include a spatial cache.  NPZ files are
accepted only when supplied externally or when a caller explicitly requests a
runtime cache in its output directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


PROBES = (2, 4, 8, 12)


def _decode(values: np.ndarray) -> np.ndarray:
    return np.asarray([value.decode() if isinstance(value, bytes) else str(value) for value in values])


def _h5ad_path(data_root: Path, probes: int) -> Path:
    candidates = (
        data_root / str(probes) / f"num{probes}_final.h5ad",
        data_root / "cell_level" / str(probes) / f"num{probes}_final.h5ad",
        data_root / f"num{probes}_final.h5ad",
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find num{probes}_final.h5ad under {data_root}; tried {candidates}")


def read_h5ad(data_root: Path, probes: int) -> tuple[np.ndarray, np.ndarray]:
    path = _h5ad_path(Path(data_root), probes)
    with h5py.File(path, "r", locking=False) as handle:
        xy = np.asarray(handle["obsm"]["spatial"])
        region_group = handle["obs"]["brain_region"]
        categories = _decode(np.asarray(region_group["categories"]))
        codes = np.asarray(region_group["codes"])
        if np.any(codes < 0) or np.any(codes >= len(categories)):
            raise ValueError(f"Invalid brain_region categorical codes in {path}")
        region = categories[codes]
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) != len(region) or not np.isfinite(xy).all():
        raise ValueError(f"Invalid spatial/brain_region arrays in {path}")
    return xy, region


def read_npz(data_root: Path, probes: int) -> tuple[np.ndarray, np.ndarray]:
    path = Path(data_root) / f"spatial_brain_region_{probes}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as loaded:
        xy = np.asarray(loaded["xy"])
        region = np.asarray(loaded["region"]).astype(str)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) != len(region) or not np.isfinite(xy).all():
        raise ValueError(f"Invalid spatial NPZ arrays in {path}")
    return xy, region


def load_spatial_inputs(data_root: Path, probes: int) -> tuple[np.ndarray, np.ndarray]:
    """Prefer current H5AD files and accept externally supplied NPZ caches."""

    root = Path(data_root)
    try:
        return read_h5ad(root, probes)
    except FileNotFoundError:
        return read_npz(root, probes)


def prepare(base_dir: Path, output_dir: Path) -> None:
    """Explicitly extract runtime NPZ caches from H5AD files."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for probes in PROBES:
        xy, region = read_h5ad(Path(base_dir), probes)
        np.savez_compressed(output_dir / f"spatial_brain_region_{probes}.npz", xy=xy, region=region)
        print(f"{probes} {len(region)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True, help="H5AD root or a directory containing external NPZ caches.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional runtime directory receiving NPZ caches.")
    args = parser.parse_args()
    if args.output_dir is not None:
        prepare(args.base_dir, args.output_dir)
    else:
        for probes in PROBES:
            xy, region = load_spatial_inputs(args.base_dir, probes)
            print(f"{probes} {len(region)}", flush=True)


if __name__ == "__main__":
    main()
