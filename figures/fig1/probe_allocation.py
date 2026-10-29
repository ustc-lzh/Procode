"""Generate Fig. 1F from the workbook's 34-cell-derived reference table."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from data_io import (
    HIGH_COLOR,
    LOW_COLOR,
    SHEET_F,
    SHEET_F_REF,
    SHEET_MODEL,
    read_sheet,
    require_columns,
)


def generate_f(workbook: Path, output_dir: Path):
    workbook = Path(workbook).resolve()
    output_dir = Path(output_dir).resolve()
    table = read_sheet(workbook, SHEET_F)
    reference = read_sheet(workbook, SHEET_F_REF)
    model = read_sheet(workbook, SHEET_MODEL)
    require_columns(table, ('gene', 'expr_for_probe_number', 'expression_group', 'probe_num'), SHEET_F)
    require_columns(reference, ('gene', 'max_cell_type_mean_expression'), SHEET_F_REF)
    require_columns(model, ('q05', 'q95', 'probe_min', 'probe_max'), SHEET_MODEL)
    if len(model) != 1:
        raise ValueError('Fig. 1F model sheet must contain one row')
    values = pd.to_numeric(reference['max_cell_type_mean_expression'], errors='raise')
    genes = reference['gene'].astype(str)
    if genes.duplicated().any():
        values = values.groupby(genes).max()
    else:
        values.index = genes
    lo, hi = float(values.quantile(0.05)), float(values.quantile(0.95))
    model_row = model.iloc[0]
    if not np.isclose(float(model_row.q05), lo) or not np.isclose(float(model_row.q95), hi):
        raise ValueError(f'Fig. 1F q05/q95 mismatch: workbook {model_row.q05}, {model_row.q95}; reference {lo}, {hi}')
    if int(model_row.probe_min) != 4 or int(model_row.probe_max) != 12:
        raise ValueError('Fig. 1F model range must remain 4-12 probes')
    table = table.copy()
    table['expr_for_probe_number'] = pd.to_numeric(table['expr_for_probe_number'], errors='raise')
    table['probe_num'] = pd.to_numeric(table['probe_num'], errors='raise').astype(int)
    expected_expr = values.reindex(table['gene'].astype(str)).to_numpy(dtype=float)
    if np.isnan(expected_expr).any() or not np.allclose(table['expr_for_probe_number'], expected_expr):
        raise ValueError('Fig. 1F gene expression values do not match the reference sheet')
    expected_probe = np.floor(np.clip(12 - (table['expr_for_probe_number'] - lo) / (hi - lo) * 8, 4, 12) + 0.5).astype(int)
    if not np.array_equal(expected_probe.to_numpy(), table['probe_num'].to_numpy()):
        raise ValueError('Fig. 1F assigned probe numbers fail the half-up model')
    colors={'High expression':HIGH_COLOR,'Low expression':LOW_COLOR}
    mpl.rcParams.update({'font.family':'Arial','font.size':8,'axes.labelsize':10,
        'xtick.labelsize':8,'ytick.labelsize':8,'axes.linewidth':.7,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.unicode_minus':False})
    fig=plt.figure(figsize=(84/25.4,82/25.4))
    ax=fig.add_axes([15/84,15/82,67/84,64/82])
    x=np.r_[0,lo,hi,42]
    ax.plot(x,[12,12,4,4],color='#F28E2B',lw=1.6,zorder=2)
    for q in (lo,hi):ax.axvline(q,color='#777777',ls=':',lw=.65,zorder=1)
    for group,col in colors.items():
        d=table[table.expression_group==group]
        ax.scatter(d.expr_for_probe_number,d.probe_num,s=29,c=col,edgecolor='white',lw=.4,zorder=3)
    offsets={
        'Opn3':(0,8,'center'), 'Fibcd1':(9,-2,'left'),
        'Arhgap25':(8,-2,'left'), 'Wfs1':(8,1,'left'),
        'Gsg1l':(8,2,'left'), 'Mapk4':(8,7,'left'),
        'Bdnf':(10,-5,'left'), 'Dkk3':(8,3,'left'),
        'Slc17a6':(0,8,'center'), 'Gad1':(-7,7,'right'), 'Cpne4':(0,18,'center')}
    annotations=[]
    for r in table.itertuples(index=False):
        dx,dy,ha=offsets[r.gene]
        annotations.append(ax.annotate(r.gene,(r.expr_for_probe_number,r.probe_num),xytext=(dx,dy),
            textcoords='offset points',fontsize=6,fontstyle='italic',ha=ha,va='center',
            arrowprops={'arrowstyle':'-','color':'#666666','lw':.35,'shrinkA':1,'shrinkB':3},zorder=4))
    ax.set_xlim(-.4,42);ax.set_ylim(3.25,12.7)
    ax.set_xticks([0,10,20,30,40]);ax.set_yticks([4,6,8,10,12])
    ax.set_xlabel('Maximum cell-type\nmean expression',fontsize=10,labelpad=3)
    ax.set_ylabel('Assigned probe number',fontsize=10,labelpad=4)
    ax.text(lo+.5,3.38,'5%',ha='left',va='bottom',fontsize=7,color='#666666')
    ax.text(hi-.5,3.38,'95%',ha='right',va='bottom',fontsize=7,color='#666666')
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(length=3,width=.7,pad=2,labelsize=8)
    ax.grid(axis='y',color='#E5E5E5',lw=.4,zorder=0)
    handles=[Line2D([],[],color='#F28E2B',lw=1.6,label='Allocation model'),
        Line2D([],[],color='none',marker='o',markersize=4,markeredgecolor='white',markerfacecolor=colors['High expression'],label='High-expression'),
        Line2D([],[],color='none',marker='o',markersize=4,markeredgecolor='white',markerfacecolor=colors['Low expression'],label='Low-expression')]
    ax.legend(handles=handles,loc='upper right',bbox_to_anchor=(.99,.985),ncol=1,
        frameon=False,fontsize=8,handlelength=1.2,labelspacing=.25,borderpad=0)
    output_dir.mkdir(parents=True,exist_ok=True)
    table.to_csv(output_dir / 'panel_F_assignment.csv', index=False, encoding='utf-8')
    for fmt in ['pdf','svg','png']:
        fig.savefig(output_dir/f'panel_F.{fmt}',dpi=300,facecolor='white')
    plt.close(fig)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--workbook',
        type=Path,
        default=Path(__file__).resolve().with_name('fig1_data.xlsx'),
        help='XLSX workbook; default is fig1_data.xlsx beside this script.',
    )
    parser.add_argument('--output-dir',type=Path,required=True,
                        help='Directory receiving panel_F.pdf, panel_F.svg and panel_F.png.')
    args=parser.parse_args()
    generate_f(args.workbook,args.output_dir)
