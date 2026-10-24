"""Independently recompute packaged predictions using NumPy/SciPy; never fits.

Optionally --data-root checks externally supplied canonical MAT file hashes and
ordered labels. Feature files are intentionally not included in S1.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results_data'
M=('ACC','NMI','PUR','ARI')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def array_hash(a):return hashlib.sha256(np.asarray(a).tobytes(order='C')).hexdigest()
def safe(name):
    p=(BASE/name).resolve();p.relative_to(ROOT.resolve());assert p.is_file();return p
def calculate(y,p):
    _,a=np.unique(y,return_inverse=True);_,b=np.unique(p,return_inverse=True)
    n=len(a);nr=int(a.max())+1;nc=int(b.max())+1
    c=np.bincount(a*nc+b,minlength=nr*nc).reshape(nr,nc)
    r,s=linear_sum_assignment(-c)
    row=[int(x) for x in c.sum(1)];col=[int(x) for x in c.sum(0)]
    mi=math.fsum(int(c[i,j])/n*math.log(int(c[i,j])*n/(row[i]*col[j])) for i in range(nr) for j in range(nc) if c[i,j])
    hy=-math.fsum(v/n*math.log(v/n) for v in row if v);hp=-math.fsum(v/n*math.log(v/n) for v in col if v)
    two=lambda v:int(v)*(int(v)-1)//2
    pairs=two(n);cells=sum(two(v) for v in c.flat);aa=sum(two(v) for v in row);bb=sum(two(v) for v in col)
    denominator=(aa+bb)*pairs-2*aa*bb
    return dict(ACC=float(c[r,s].sum()/n),NMI=float(2*mi/(hy+hp) if hy+hp else 1),
        PUR=float(c.max(0).sum()/n),ARI=float(2*(cells*pairs-aa*bb)/denominator if denominator else 1))
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data-root',type=Path);args=parser.parse_args()
    index=json.loads((BASE/'benchmark_reproductions/run_index.json').read_text(encoding='utf-8'))
    records=index['targeted_runs']+index['extra_native_predictions']+index['own_bound_predictions']
    result=[];values_by_id={};normalized_labels={}
    for record in records:
        path=safe(record['prediction_file']);assert sha(path)==record['prediction_sha256']
        with np.load(path,allow_pickle=False) as data:
            y=np.asarray(data['y_true']).ravel();p=np.asarray(data['y_pred']).ravel();order=np.asarray(data['sample_order_index_zero_based']).ravel()
            assert len(y)==len(p)==record['samples'] and len(np.unique(y))==record['classes']
            assert np.isfinite(y).all() and np.isfinite(p).all()
            assert np.array_equal(y,np.floor(y)) and np.array_equal(p,np.floor(p))
            assert np.array_equal(order,np.arange(len(y)))
            for key,meta in record['arrays'].items():
                array=np.asarray(data[key]);assert array.dtype.str==meta['dtype'] and list(array.shape)==meta['shape']
                assert array_hash(array)==meta['sha256_data_C']
        _,normalized=np.unique(y,return_inverse=True)
        expected=index['input_files'][record['dataset']]['labels_ordered_sha256']
        assert hashlib.sha256(np.asarray(normalized,dtype='<i8').tobytes()).hexdigest()==expected
        normalized_labels[record['dataset']]=normalized
        values=calculate(y,p);values_by_id[record['run_id']]=values
        if record.get('metrics_display_precision_only'):
            assert all(f'{values[k]:.4f}'==f'{record["metrics"][k]:.4f}' for k in M)
        else:assert max(abs(values[k]-record['metrics'][k]) for k in M)<1e-12,(record['run_id'],values,record['metrics'])
        result.append(dict(run_id=record['run_id'],method=record['method'],dataset=record['dataset'],metrics=values,
            max_absolute_metric_delta=max(abs(values[k]-record['metrics'][k]) for k in M),
            precision_comparison='display_4dp' if record.get('metrics_display_precision_only') else 'full_precision'))
    for selection in index['selections']:
        best=max(selection['available_run_ids'],key=lambda key:(values_by_id[key]['ACC'],-int(key.rsplit('|',1)[1])))
        assert best==selection['selected_run_id']
        assert all(abs(values_by_id[best][k]-selection['metrics'][k])<1e-12 for k in M)
    input_checks=[]
    if args.data_root:
        from scipy.io import loadmat
        for dataset,meta in index['input_files'].items():
            path=args.data_root/meta['file_name'];assert sha(path)==meta['input_sha256'],dataset
            data=loadmat(path,variable_names=['Y','truelabel'])
            label=data['Y'] if 'Y' in data else data['truelabel']
            while np.asarray(label).dtype==object:label=np.asarray(label).ravel()[0]
            _,label=np.unique(np.asarray(label).ravel(),return_inverse=True)
            assert np.array_equal(label,normalized_labels[dataset]),dataset
            input_checks.append(dataset)
    print(json.dumps(dict(status='PASS',targeted_partitions=31,RAV_native_partitions=5,own_RGBD_bound_partition=1,
        own_other_main_entries='five receipt/log-supported entries; not prediction-recomputed',
        NMI='explicit arithmetic entropy normalization',metric_implementation='independent integer confusion/Hungarian, explicit MI/entropy and pair-count ARI',
        best_complete_run_selections_verified=5,external_canonical_inputs_checked=input_checks,
        data_root_not_supplied_boundary='input file hashes and ordered labels are recorded; original MAT payloads are not in S1' if not args.data_root else '',
        model_imports=0,training_or_inference_or_KMeans_fit=0,partitions=result),indent=2))
if __name__=='__main__':main()
