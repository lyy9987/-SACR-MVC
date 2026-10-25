"""Replot Figure 6 from saved representations. This does not train a model.

Portable, RGB-D-only adaptation of make_tsne_figure_v18.py. The original
stage files and settings are retained. Requires numpy, matplotlib, sklearn.
"""
import argparse
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=BASE/'reproduced',
                        help="Directory for the Figure 6 PDF, PNG and coordinates")
    args=parser.parse_args(argv)
    import csv
    import inspect
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    with np.load(BASE/'visualization/RGB-D_stage0.npz',allow_pickle=False) as z:
        stages={0:z['stage0']}
    with np.load(BASE/'visualization/RGB-D_finetune.npz',allow_pickle=False) as z:
        stages.update({int(k.replace('stage','')):z[k] for k in z.files})
    with (BASE/'visualization/RGB-D_labels.csv').open(encoding='utf-8',newline='') as f:
        rows=list(csv.DictReader(f))
    assert [int(r['sample_index']) for r in rows]==list(range(1449))
    truth=np.array([int(r['class_label']) for r in rows],dtype=int)
    keys=(0,25,50,100)
    rng=np.random.default_rng(0)
    index=np.sort(rng.choice(len(truth),size=min(2500,len(truth)),replace=False))
    plt.rcParams.update({'font.size':9,'text.color':'black','axes.unicode_minus':False,
                         'figure.facecolor':'white','pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,4,figsize=(7.2,1.85))
    coordinates={}
    for i,(ax,stage) in enumerate(zip(axes,keys)):
        kwargs=dict(n_components=2,random_state=0,perplexity=30,learning_rate=200.0,init='pca')
        kwargs['max_iter' if 'max_iter' in inspect.signature(TSNE.__init__).parameters else 'n_iter']=2000
        emb=TSNE(**kwargs).fit_transform(stages[stage][index])
        coordinates[f'epoch{stage}']=emb
        ax.scatter(emb[:,0],emb[:,1],c=truth[index],cmap='tab20',s=4.5,alpha=.9,linewidths=0)
        ax.set_axis_off();ax.set_aspect('equal',adjustable='datalim')
        ax.text(.5,-.06,f'({"abcd"[i]}) Epoch {stage}',transform=ax.transAxes,ha='center',va='top',fontsize=9)
    fig.subplots_adjust(left=.005,right=.995,top=.995,bottom=.12,wspace=.03)
    output=args.output.resolve()
    output.mkdir(exist_ok=True)
    for ext in ('png','pdf'):
        fig.savefig(output/('Fig6_RGBD.'+ext),dpi=300,bbox_inches='tight')
    np.savez_compressed(output/'Fig6_RGBD_coordinates.npz',sample_index=index,labels=truth[index],**coordinates)
    print(output/'Fig6_RGBD.pdf')


if __name__=='__main__':
    main()
