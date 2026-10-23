"""Copy original Result1 H5AD files unchanged and audit regional count totals.

The four supplied inputs contain cell-level counts and cell-centroid spatial
coordinates, not the large individual-transcript table used by the separate
nearest-neighbour analysis. No H5AD fields are removed or modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from data_io import PROBES, SHEET_C, SHEET_E, SHEET_REGIONS, read_sheet

OBS_FIELDS = ('cell_id', 'pixel_area', 'brain_region', 'fu_region', 'fov')
EXCLUDED_QC_GENES = {'blank', 'Foxp2', 'Lamp5'}


def read_column(node):
    if isinstance(node, h5py.Group):
        codes = node['codes'][:]
        categories = read_column(node['categories'])
        return np.array([categories[i] if i >= 0 else None for i in codes], dtype=object)
    return node.asstr()[:] if h5py.check_string_dtype(node.dtype) else node[:]


def read_counts(handle):
    x = handle['X']
    if isinstance(x, h5py.Dataset):
        return sparse.csr_matrix(x[:])
    encoding = x.attrs['encoding-type']
    if isinstance(encoding, bytes):
        encoding = encoding.decode()
    constructor = {'csr_matrix': sparse.csr_matrix, 'csc_matrix': sparse.csc_matrix}[encoding]
    return constructor((x['data'][:], x['indices'][:], x['indptr'][:]), shape=tuple(x.attrs['shape'])).tocsr()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _source_h5ad(base_dir: Path, probes: int) -> Path:
    candidates = (
        base_dir / str(probes) / f'num{probes}_final.h5ad',
        base_dir / 'cell_level' / str(probes) / f'num{probes}_final.h5ad',
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f'Could not find num{probes}_final.h5ad under {base_dir}; tried {candidates}')


def export(base_dir, output_dir, workbook):
    base_dir, output_dir, workbook = map(Path, (base_dir, output_dir, workbook))
    definitions = read_sheet(workbook, SHEET_REGIONS)
    line = read_sheet(workbook, SHEET_C)
    quadrant = read_sheet(workbook, SHEET_E)
    reports, aggregates = [], []
    for probes in PROBES:
        src = _source_h5ad(base_dir, probes)
        dst = output_dir / 'cell_level' / str(probes) / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            raise FileExistsError(dst)
        with h5py.File(src, 'r', locking=False) as original:
            for key in OBS_FIELDS:
                if key not in original['obs']:
                    raise KeyError(f'{src}: required obs/{key} absent')
            obs_index = original['obs'].attrs['_index']
            var_index = original['var'].attrs['_index']
            counts = read_counts(original)
            if not np.isfinite(counts.data).all() or (counts.data < 0).any() or not np.equal(counts.data, np.floor(counts.data)).all():
                raise ValueError(f'{src}: X is not nonnegative raw integer counts')
            genes = read_column(original['var'][var_index]).astype(str)
            if len(set(genes)) != len(genes):
                raise ValueError('Duplicate gene names require explicit upstream disambiguation')
            cell_ids = read_column(original['obs'][obs_index])
            if len(set(cell_ids)) != len(cell_ids):
                raise ValueError('Cell index is not unique')
            region = read_column(original['obs']['brain_region'])
            area = np.asarray(original['obs']['pixel_area'][:], dtype=float)
            xy = original['obsm']['spatial'][:]
            if xy.shape != (counts.shape[0], 2) or not np.isfinite(xy).all():
                raise ValueError('Expected finite 2-D cell-centroid coordinates')
            retained_genes = np.array([g not in EXCLUDED_QC_GENES for g in genes])
            totals = np.asarray(counts[:, retained_genes].sum(axis=1)).ravel()
            keep = (totals > 1) & (totals < 100) & (area > 1000) & (area < 20000)
            # Audit this known input layout, rather than silently accepting a
            # future H5AD with an attached molecule-coordinate cache.
            if set(original['obsm']) != {'spatial'} or set(original['uns']) != {'fu_region_colors'} or len(original['layers']):
                raise ValueError('Unexpected H5AD payload; inspect its coordinate/uns contents before copying')
            original_obs_fields = set(original['obs'])
        shutil.copy2(src, dst)
        with h5py.File(dst, 'r') as released:
            assert set(released['obs']) == original_obs_fields
            assert set(released['obsm']) == {'spatial'}
            assert set(released['uns']) == {'fu_region_colors'} and len(released['layers']) == 0
            reconstructed = read_counts(released)
            assert (reconstructed != counts).nnz == 0
            assert np.array_equal(read_column(released['obs'][obs_index]), cell_ids)
            assert np.array_equal(released['obsm']['spatial'][:], xy)
        for brain_region in sorted(set(region[keep]) - {None}):
            mask = keep & (region == brain_region)
            matrix = counts[mask]
            gene_totals = np.asarray(matrix.sum(axis=0)).ravel()
            positives = np.asarray((matrix > 0).sum(axis=0)).ravel()
            for j, gene in enumerate(genes):
                if retained_genes[j]:
                    aggregates.append({'probes': probes, 'brain_region': brain_region, 'gene': gene,
                                       'n_cells': int(mask.sum()), 'total_counts': int(gene_totals[j]),
                                       'n_positive_cells': int(positives[j]),
                                       'mean_counts_per_cell': float(gene_totals[j] / mask.sum())})
        # The two archived figure source tables must be supported by this X.
        for row in line[line.probes == probes].itertuples(index=False):
            main = definitions[(definitions.gene == row.gene) & (definitions.region_set == 'allen_top3_main_expression')].region
            negative = definitions[(definitions.gene == row.gene) & (definitions.region_set == 'reference_negative')].region
            selected = keep & np.isin(region, main)
            gi, gap = np.where(genes == row.gene)[0][0], np.where(genes == 'Gapdh')[0][0]
            actual = (float(counts[selected, gi].sum()), float(counts[selected, gap].sum()), int(selected.sum()))
            expected = (row.gene_total_spots, row.gapdh_total_spots, row.n_selected_cells)
            if not np.allclose(actual, expected, rtol=0, atol=1e-8):
                raise ValueError(f'{probes}/{row.gene}: archived signal totals do not match {actual} != {expected}')
            q = quadrant[(quadrant.probes == probes) & (quadrant.gene == row.gene)].iloc[0]
            neg_total = float(counts[keep & np.isin(region, negative), gi].sum())
            all_total = float(counts[keep & pd.notna(region), gi].sum())
            if not np.allclose([neg_total, all_total], [q.false_region_total_spots, q.all_region_total_spots], rtol=0, atol=1e-8):
                raise ValueError(f'{probes}/{row.gene}: archived negative-region totals do not match')
        report = {'probes': probes, 'n_input_cells': int(counts.shape[0]), 'n_genes': int(counts.shape[1]),
                  'n_qc_retained_cells': int(keep.sum()), 'X_total_counts': int(counts.sum()),
                  'file': dst.relative_to(output_dir).as_posix(), 'source_sha256': sha(src), 'export_sha256': sha(dst),
                  'signal_and_reference_negative_totals_match_archived_tables': True,
                  'contains_transcript_coordinates': False, 'contains_cell_centroids': True,
                  'copy_method': 'original H5AD copied byte-for-byte; no fields removed'}
        assert report['source_sha256'] == report['export_sha256']
        reports.append(report)
        print(f'num{probes}: {counts.shape[0]} cells, QC {keep.sum()}, source tables matched', flush=True)
    pd.DataFrame(aggregates).to_csv(output_dir / 'regional_expression_counts.csv', index=False)
    pd.DataFrame(reports).to_csv(output_dir / 'cell_level_manifest.csv', index=False)
    (output_dir / 'cell_level_validation.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir', type=Path, required=True, help='Original 2/4/8/12 experiment directories')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--workbook', type=Path, default=Path(__file__).resolve().with_name('fig1_data.xlsx'))
    args = parser.parse_args()
    export(args.base_dir, args.output_dir, args.workbook)
