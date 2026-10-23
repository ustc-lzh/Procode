"""Plot SFig. 2A-C from cell-centroid H5AD spatial inputs."""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

try:
    from .nearest_neighbour_plot import plot_panel_d
    from .spatial_inputs import load_spatial_inputs
except ImportError:  # direct execution from the supplementary folder
    from nearest_neighbour_plot import plot_panel_d
    from spatial_inputs import load_spatial_inputs

try:
    from data_io import SHEET_C, SHEET_D, SHEET_REGIONS, read_sheet
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data_io import SHEET_C, SHEET_D, SHEET_REGIONS, read_sheet

ROOT=Path(__file__).resolve().parent
PROBES=(2,4,8,12)
LOW='#00AEEF';HIGH='#ED1C24'
REGION_COLORS={'OLF':'#43A756','Isocortex':'#EE9999','TH_ex':'#8882BD',
    'TH_RT':'#373486','MB':'#92C981','CB':'#347CB4','STR':'#E34A33',
    'HPF':'#E78BC1','TH_AV':'#C5BBDE','HY':'#F1B46D','HB':'#9B7768',
    'IP':'#79B8D8','fiber':'#D6E4EE','other':'#F1F1F1'}
CLASS_COLORS={'Main-expression regions':HIGH,'Reference-negative regions':LOW,
              'Other regions':'#DEDEDE'}
def style(font_dir=None):
    if font_dir is not None:
        for name in ['arial.ttf','arialbd.ttf','ariali.ttf','arialbi.ttf']:
            font_manager.fontManager.addfont(str(font_dir/name))
    mpl.rcParams.update({'font.family':'DejaVu Sans' if font_dir is None else 'Arial','font.size':8,'axes.labelsize':10,
        'axes.titlesize':8,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
        'axes.linewidth':.7,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
        'mathtext.fontset':'dejavusans' if font_dir is None else 'custom','mathtext.rm':'Arial','mathtext.it':'Arial:italic',
        'mathtext.bf':'Arial:bold','savefig.dpi':600})


def load_inputs(data_root, workbook):
    datasets={p:dict(zip(('xy', 'region'), load_spatial_inputs(Path(data_root), p))) for p in PROBES}
    defs=read_sheet(Path(workbook), SHEET_REGIONS)
    assert defs.annotation_column.eq('brain_region').all()
    expected={'Slc17a6':{'main':['TH_AV','TH_ex','MB','HY'],'negative':['HPF','STR','fiber']},
        'Opn3':{'main':['CB','TH_ex','Isocortex','IP'],'negative':['HPF','MB','HB','fiber']}}
    rules={}
    for gene,definition in expected.items():
        rules[gene]={}
        for name,category in [('main','allen_top3_main_expression'),('negative','reference_negative')]:
            actual=defs[(defs.gene==gene)&(defs.region_set==category)].sort_values('order').region.tolist()
            assert actual==definition[name],(gene,name,actual)
            rules[gene][name]=actual
        assert not set(rules[gene]['main'])&set(rules[gene]['negative'])
    for p,data in datasets.items():
        assert set(data['region'])<=set(REGION_COLORS),(p,set(data['region']))
        for gene in rules:
            assert set(rules[gene]['main']+rules[gene]['negative'])<=set(data['region'])
    return datasets,rules


def spatial_axis(ax,xy):
    # Same coordinate orientation and pixel-to-distance calibration as source code.
    lo=xy.min(axis=0);hi=xy.max(axis=0);pad=(hi-lo)*.045
    ax.set_xlim(lo[0]-pad[0],hi[0]+pad[0]);ax.set_ylim(lo[1]-pad[1],hi[1]+pad[1])
    ax.set_aspect('equal',adjustable='box',anchor='N')
    ax.set_xticks([]);ax.set_yticks([])
    for s in ax.spines.values():s.set_visible(False)
    x0,x1=ax.get_xlim();y0,y1=ax.get_ylim()
    xe=x1-.055*(x1-x0);xs=xe-1000/.1625;y=y0+.035*(y1-y0)
    ax.plot([xs,xe],[y,y],color='black',lw=.8,solid_capstyle='butt')
    ax.annotate('1 mm',xy=((xs+xe)/2,y),xytext=(0,-2),textcoords='offset points',
                ha='center',va='top',fontsize=6,annotation_clip=False)


def save_panel(fig,letter,panel_dir):
    folder=Path(panel_dir);folder.mkdir(parents=True,exist_ok=True)
    fig.savefig(folder/f'panel_{letter}.pdf',facecolor='white')
    fig.savefig(folder/f'panel_{letter}.svg',facecolor='white')
    fig.savefig(folder/f'panel_{letter}.png',dpi=180,facecolor='white')
    plt.close(fig)


def panel_a(datasets,panel_dir):
    fig=plt.figure(figsize=(195/25.4,52/25.4))
    for i,p in enumerate(PROBES):
        ax=fig.add_axes([.006+i*.247,.28,.244,.59])
        xy=datasets[p]['xy'];region=datasets[p]['region']
        for name,color in REGION_COLORS.items():
            mask=region==name
            ax.scatter(xy[mask,0],xy[mask,1],s=.15,c=color,alpha=.92,linewidths=0,rasterized=True)
        spatial_axis(ax,xy)
        ax.set_title(f'Brain region | {p} probes',fontsize=8,fontweight='bold',pad=4)
    handles=[Line2D([],[],marker='o',linestyle='none',markerfacecolor=c,markeredgecolor='#555555',
        markeredgewidth=.2,markersize=4,label=n) for n,c in REGION_COLORS.items()]
    fig.legend(handles=handles,title='Brain region',title_fontsize=8,fontsize=8,ncol=7,
        loc='lower center',bbox_to_anchor=(.5,.002),frameon=False,handletextpad=.25,
        columnspacing=.9,labelspacing=.35)
    save_panel(fig,'A',panel_dir)


def panel_b(datasets,rules,output_dir,panel_dir):
    fig=plt.figure(figsize=(195/25.4,112/25.4));audit=[]
    for row,gene in enumerate(('Slc17a6','Opn3')):
        top=.98-row*.475;bottom=.565-row*.475
        fig.text(.018,top,gene,fontstyle='italic',fontsize=10,va='top')
        fig.text(.20,top,'Main-expression: '+', '.join(rules[gene]['main']),fontsize=8,va='top')
        fig.text(.20,top-.038,'Reference-negative: '+', '.join(rules[gene]['negative']),fontsize=8,va='top')
        for i,p in enumerate(PROBES):
            ax=fig.add_axes([.006+i*.247,bottom,.244,.31])
            xy=datasets[p]['xy'];region=datasets[p]['region']
            classes=np.full(len(region),'Other regions',dtype='<U30')
            classes[np.isin(region,rules[gene]['negative'])]='Reference-negative regions'
            classes[np.isin(region,rules[gene]['main'])]='Main-expression regions'
            for label in ['Other regions','Reference-negative regions','Main-expression regions']:
                mask=classes==label
                ax.scatter(xy[mask,0],xy[mask,1],s=.13 if label=='Other regions' else .17,
                    c=CLASS_COLORS[label],alpha=.30 if label=='Other regions' else .96,
                    linewidths=0,rasterized=True)
                audit.append({'gene':gene,'probes':p,'category':label,'n_cells':int(mask.sum()),
                    'regions':','.join(sorted(np.unique(region[mask]))),'annotation_column':'brain_region'})
            spatial_axis(ax,xy);ax.set_title(f'{p} probes',fontsize=8,fontweight='bold',pad=4)
    handles=[Line2D([],[],marker='o',linestyle='none',markerfacecolor=c,markeredgecolor='none',
        markersize=4,label=n) for n,c in CLASS_COLORS.items()]
    fig.legend(handles=handles,ncol=3,loc='lower center',bbox_to_anchor=(.5,.005),
        frameon=False,fontsize=8,columnspacing=1.2,handletextpad=.35)
    pd.DataFrame(audit).to_csv(Path(output_dir)/'panel_B_classification.csv',index=False)
    save_panel(fig,'B',panel_dir)


def panel_c(workbook,output_dir,panel_dir):
    frame=read_sheet(Path(workbook), SHEET_C)
    data=frame[frame.gene.isin(['Slc17a6','Opn3'])].copy()
    assert len(data)==8 and not data.duplicated(['gene','probes']).any()
    assert np.allclose(data.gene_total_spots/data.gapdh_total_spots,data.gene_gapdh_total_ratio)
    assert np.allclose(data.gene_gapdh_total_ratio/data.baseline_ratio,data.ratio_vs_num2)
    data.to_csv(Path(output_dir)/'panel_C_values.csv',index=False)
    fig=plt.figure(figsize=(94/25.4,84/25.4));ax=fig.add_axes([19/94,16/84,72/94,64/84])
    for gene,color,group in [('Slc17a6',HIGH,'high'),('Opn3',LOW,'low')]:
        sub=data[data.gene==gene].sort_values('probes')
        assert sub.probes.tolist()==list(PROBES)
        ax.plot(sub.probes,sub.ratio_vs_num2,marker='o',ms=3.5,lw=1.4,color=color,
            markeredgecolor='white',markeredgewidth=.4,label=rf'$\it{{{gene}}}$ ({group})')
    maximum=float(data.ratio_vs_num2.max());upper=float(np.ceil(maximum*1.08))
    ax.axhline(1,color='#AAAAAA',lw=.7,linestyle=(0,(2,2)),zorder=0)
    ax.set_xlim(1.5,12.5);ax.set_ylim(.7,upper);ax.set_xticks(PROBES)
    ax.set_yticks([1]+list(np.arange(2,upper+.1,2)))
    ax.set_xlabel('Probe number');ax.set_ylabel('Normalized signal gain\n(fold vs. 2 probes)')
    ax.text(.5,1.015,'Main-expression regions',transform=ax.transAxes,fontsize=8,
        ha='center',va='bottom',color='#333333')
    ax.legend(loc='upper left',frameon=False,borderaxespad=.25,handlelength=1.7)
    for spine in ('top','right'):ax.spines[spine].set_visible(False)
    ax.tick_params(width=.7,length=3,pad=2)
    save_panel(fig,'C',panel_dir)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True,
                   help='H5AD root (or externally supplied spatial NPZ cache directory).')
    p.add_argument('--workbook',type=Path,default=ROOT.parent/'fig1_data.xlsx',
                   help='XLSX workbook containing region and summary sheets.')
    p.add_argument('--output-dir',type=Path,required=True,
                   help='Directory receiving SFig2 plots and summary tables.')
    p.add_argument('--font-dir',type=Path,default=None,
                   help='Optional directory containing Arial font files.')
    args=p.parse_args()
    output_dir=args.output_dir.resolve();panel_dir=output_dir/'panels'
    style(args.font_dir);datasets,rules=load_inputs(args.data_root,args.workbook)
    panel_a(datasets,panel_dir);print('A complete',flush=True)
    panel_b(datasets,rules,output_dir,panel_dir);print('B complete',flush=True)
    panel_c(args.workbook,output_dir,panel_dir);print('C complete',flush=True)
    plot_panel_d(read_sheet(args.workbook, SHEET_D),panel_dir/'panel_D.pdf')
    print('D complete',flush=True)


if __name__=='__main__':main()
