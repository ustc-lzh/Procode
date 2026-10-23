"""Recompute Fig. 1 C/E from QC-passing regional count totals."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from data_io import SHEET_C, SHEET_E, SHEET_REGIONS, read_sheet


def recompute(regional_counts, workbook, output_dir):
    workbook, output_dir = Path(workbook).resolve(), Path(output_dir).resolve()
    data = pd.read_csv(regional_counts)
    definitions = read_sheet(workbook, SHEET_REGIONS)
    archived = read_sheet(workbook, SHEET_E)
    line = read_sheet(workbook, SHEET_C)
    if set(line['gene'].astype(str)) != set(archived['gene'].astype(str)):
        raise ValueError('Fig. 1 C/E workbook gene sets differ')
    if data.duplicated(['probes', 'brain_region', 'gene']).any():
        raise ValueError('Duplicate regional count keys')
    rows = []
    for reference in archived.itertuples(index=False):
        current = data[(data.probes == reference.probes)]
        rules = definitions[definitions.gene == reference.gene]
        main = rules[rules.region_set == 'allen_top3_main_expression'].region
        negative = rules[rules.region_set == 'reference_negative'].region
        gene = current[current.gene == reference.gene]
        gapdh = current[current.gene == 'Gapdh']
        gene_main = gene[gene.brain_region.isin(main)].total_counts.sum()
        gapdh_main = gapdh[gapdh.brain_region.isin(main)].total_counts.sum()
        negative_total = gene[gene.brain_region.isin(negative)].total_counts.sum()
        all_total = gene.total_counts.sum()
        if gapdh_main <= 0 or all_total <= 0:
            raise ValueError('Nonpositive denominator')
        rows.append({'gene': reference.gene, 'probes': int(reference.probes), 'region_mode': 'top3',
                     'group': reference.group, 'gene_total_spots': int(gene_main), 'gapdh_total_spots': int(gapdh_main),
                     'n_selected_cells': int(gene[gene.brain_region.isin(main)].n_cells.sum()),
                     'gene_gapdh_total_ratio': gene_main / gapdh_main,
                     'false_region_total_spots': int(negative_total), 'all_region_total_spots': int(all_total),
                     'false_ratio': negative_total / all_total})
    result = pd.DataFrame(rows)
    baseline = result[result.probes == 2].set_index('gene')
    result['baseline_ratio'] = result.gene.map(baseline.gene_gapdh_total_ratio)
    result['baseline_false_ratio'] = result.gene.map(baseline.false_ratio)
    result['x_ratio'] = result.gene_gapdh_total_ratio / result.baseline_ratio
    # Preserve the historical source column explicitly, and separate its sign
    # from the current E display rather than silently rewriting archived input.
    result['y_log2'] = -np.log2((result.false_ratio + 1e-12) / (result.baseline_false_ratio + 1e-12))
    result['y_display'] = np.log2(result.false_ratio / result.baseline_false_ratio)
    for name in ('gene_total_spots', 'gapdh_total_spots', 'n_selected_cells', 'gene_gapdh_total_ratio',
                 'false_region_total_spots', 'all_region_total_spots', 'false_ratio', 'baseline_ratio',
                 'baseline_false_ratio', 'x_ratio', 'y_log2'):
        if not np.allclose(result[name], archived[name], rtol=1e-9, atol=1e-10):
            raise ValueError(f'Archived {name} cannot be reproduced')
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / 'regional_response_recomputed.csv', index=False)
    report = {'rows': len(result), 'archived_numerical_columns_matched': True,
              'display_y': 'log2(false_ratio / baseline_false_ratio)',
              'transcript_coordinates_required': False}
    (output_dir / 'regional_response_validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--regional-counts', type=Path, required=True)
    parser.add_argument('--workbook', type=Path, default=Path(__file__).resolve().with_name('fig1_data.xlsx'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    recompute(args.regional_counts, args.workbook, args.output_dir)
