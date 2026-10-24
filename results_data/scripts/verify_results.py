"""Self-contained stdlib verification of S1 tables, sources and file integrity.

This checks receipts, atomic excerpts, recorded metrics, complete-run selection
and hashes. Optional NumPy/SciPy array recomputation is a separate script.
"""
from pathlib import Path
from decimal import Decimal
from collections import Counter
import csv
import hashlib
import json
import re

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results_data'
M=('ACC','NMI','PUR','ARI')
FIXED={'MFLVC','GCFAgg','SCM','SCMVC','DCMVC','TMCN','GDR-CGDR','RAV','CAFE','GDCN'}

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def safe(name):
    p=(BASE/name.split('#',1)[0]).resolve()
    try:p.relative_to(ROOT.resolve())
    except ValueError:raise AssertionError(name)
    assert p.is_file(),name
    return p
def match(values,row):
    for k in M:
        if row[k]!='':assert f'{float(values[k]):.4f}'==f'{float(row[k]):.4f}',(row.get('method'),row.get('dataset'),k)
def follow(obj,path):
    for k in path:obj=obj[k]
    return obj
def text_metrics(text):
    out={}
    for k in M:
        v=re.search(r'\b'+k+r'\s*[:=]\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)',text)
        if v:out[k]=float(v.group(1))
    return out

manifest=read(BASE/'SOURCE_MANIFEST.json')
checks={}
for line in (ROOT/'CHECKSUMS.sha256').read_text(encoding='utf-8').splitlines():
    expected,name=line.split('  ',1)
    assert name not in checks
    p=(ROOT/name).resolve();p.relative_to(ROOT.resolve())
    assert p.is_file() and sha(p)==expected,name
    checks[name]=expected
actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and p.name!='CHECKSUMS.sha256'}
assert set(checks)==actual,(set(checks)-actual,actual-set(checks))
for record in manifest['protected_base_files']:
    assert sha(ROOT/record['path'])==record['sha256'],record['path']

main=rows(BASE/'tables/main_comparison.csv')
assert sha(BASE/'tables/main_comparison.csv')==manifest['authoritative_main_csv_sha256']
sources=rows(BASE/'tables/entry_sources.csv')
assert len(main)==len(sources)==66
lookup={(r['method_key'],r['dataset']):r for r in main}
entry={(r['method'],r['dataset']):r for r in sources}
assert len(lookup)==len(entry)==66 and set(lookup)==set(entry)
assert {r['method_key'] for r in main}==FIXED|{'SACR-MVC'}
assert all(sum(r['method_key']==method for r in main)==6 for method in FIXED|{'SACR-MVC'})
assert all(r['table']=='3' for r in main)
for r in main:
    e=entry[r['method_key'],r['dataset']]
    assert all(r[k]==e[k] for k in M+('source_kind','statistic_type','NMI_display_comparable','NMI_source_reported'))
    assert r['origin_source']==e['origin_source']
    assert r['evidence_kind']==r['source_kind']
    for k in M:
        if r[k]!='':assert -1<=float(r[k])<=1
    if not r['origin_source'].startswith('http'):safe(r['origin_source'])
kind_counts=Counter(r['source_kind'] for r in main)
assert kind_counts==Counter({'primary_paper_reported':15,'secondary_paper_reported':6,
    'retained_local_result_record':21,'saved_final_prediction_single_run':6,
    'saved_native_prediction_single_run':5,'best_ACC_complete_epoch':2,
    'best_ACC_complete_run_from_5':5,'own_retained_run':6}),kind_counts
for d,v in [('ALOI-100','0.8355'),('Hdigit','0.9770')]:
    r=lookup['GDR-CGDR',d];assert r['NMI']=='' and r['NMI_source_reported']==v and r['NMI_display_comparable']=='false'
stats=read(BASE/'evidence/source_statistics_and_candidates.json')
for group in stats['GCFAgg_candidates']['groups']:
    candidates=[x['record'] for x in stats['GCFAgg_candidates']['records'] if x['record']['dataset']==group['dataset']]
    assert len(candidates)==3
    best=max(candidates,key=lambda x:float(x['ACC']))
    r=lookup['GCFAgg',group['dataset']];match(best,r)
    assert r['available_run_count']=='3' and r['selected_run_count']=='1' and r['selected_seed']==str(best['seed'])
for info in stats['published_source_statistics']:
    r=lookup[info['method'],info['dataset']]
    assert r['statistic_type']==info['statistic_type']
    assert r['source_statistic_metadata'].startswith('evidence/source_statistics_and_candidates.json#')

fig=rows(BASE/'figures/fig4_acc.csv')
assert len(fig)==66
for r in fig:
    original=lookup[r['method_key'],r['dataset']]
    assert Decimal(r['ACC_percent'])==Decimal(original['ACC'])*100
    assert r['method_order']==original['method_order']

receipt={}
for line in (BASE/'receipts/frozen_selected.jsonl').read_text(encoding='utf-8').splitlines():
    o=json.loads(line);assert o['exit_code']==0 and o['status']=='completed';receipt[o['id']]=o
for p in (BASE/'receipts').glob('sacr_*.json'):
    o=read(p);assert o['exit_code']==0;receipt[o['id']]=o
line=(BASE/'receipts/cifar10_promoted_RESULT_JSON.txt').read_text(encoding='utf-8').strip()
receipt['cifar10_r5_svd384_tw20_seed2424']=json.loads(line[len('RESULT_JSON '):])
def receipt_values(obj):
    if 'result' in obj:return obj['result']['primary_metrics']
    return obj['metrics']
for e in sources:
    if e['source_kind']=='own_retained_run':
        obj=receipt[e['receipt_id']]
        match(follow(obj,json.loads(e['metric_object_path_json'])),e)
excerpts={}
for line in (BASE/'evidence/entry_excerpts.jsonl').read_text(encoding='utf-8').splitlines():
    o=json.loads(line);assert o['entry_id'] not in excerpts;excerpts[o['entry_id']]=o
    assert hashlib.sha256(o['exact_line'].encode('utf-8')).hexdigest()==o['exact_line_utf8_sha256']
assert len(excerpts)==21
for e in sources:
    if e['source_kind']=='retained_local_result_record':
        o=excerpts[e['entry_id']]
        values=follow(json.loads(o['exact_line']),o['metric_object_path']) if o['record_type']=='json' else text_metrics(o['exact_line'])
        match(values,e)
        assert o['source_file_sha256']==e['source_file_sha256']
        assert o['exact_line_utf8_sha256']==e['source_record_sha256']
        if o.get('supporting_native_final_log'):
            native=o['supporting_native_final_log']
            assert hashlib.sha256(native['exact_line'].encode('utf-8')).hexdigest()==native['exact_line_utf8_sha256']
            assert native['complete_evaluation_vectors_in_log']==1
            match(text_metrics(native['exact_line']),e)

index=read(BASE/'benchmark_reproductions/run_index.json')
targeted=index['targeted_runs'];extra=index['extra_native_predictions'];own=index['own_bound_predictions']
assert len(targeted)==31 and len(extra)==5 and len(own)==1
assert Counter(v['method'] for v in targeted)==Counter({'GDR-CGDR':20,'TMCN':5,'SCM':6})
run_map={v['run_id']:v for v in targeted+extra+own};assert len(run_map)==37
for v in run_map.values():
    assert sha(safe(v['prediction_file']))==v['prediction_sha256']
    for key in ['config_file','source_record_file','input_manifest_file']:
        if v.get(key):assert sha(safe(v[key]))==v[key.replace('_file','_sha256')]
    if v.get('source_record_file'):
        obj=read(safe(v['source_record_file']));values=obj['metrics']
        if v['method']=='GDR-CGDR':values=dict(ACC=values['ACC'],NMI=values['NMI_arithmetic_shared'],PUR=values['PUR'],ARI=values['ARI'])
        assert all(abs(values[k]-v['metrics'][k])<1e-12 for k in M)
    if v.get('config_file'):
        cfg=read(safe(v['config_file']))
        if v['method']=='GDR-CGDR':
            assert cfg['h']==(10 if v['dataset'] in ('RGB-D','Mfeat') else 12) and cfg['k']==20
            assert cfg['init_label']=='KM' and cfg['Maxiter']==30 and cfg['independent_correct_onehot_semantic_patch']
            assert not cfg.get('cross_seed_graph_cache_used',False)
        elif v['method']=='TMCN':assert cfg['total_epochs']==300 and cfg['checkpoint']=='final_only'
        elif v['method']=='SCM':
            for k,expected in {'training_seed':1,'training_runs':1,'iterations':1000,'batch_size':256,'feature_dim':256,
                               'high_feature_dim':128,'learning_rate':0.0003,'KMeans_n_init':10,'KMeans_random_state':1}.items():assert cfg[k]==expected
    if v.get('receipt_binding_id'):
        bound=receipt[v['receipt_binding_id']]['result']
        assert bound['route']['prediction_sha256']==v['historical_array_sha256']==v['arrays']['y_pred']['sha256_data_C']
        assert all(abs(bound['primary_metrics'][k]-v['metrics'][k])<1e-12 for k in M)
for s in index['selections']:
    rs=[run_map[k] for k in s['available_run_ids']];assert len(rs)==5
    assert sorted(v['seed'] for v in rs)==([0,1,2,3,4] if s['method']=='TMCN' else [1,2,3,4,5])
    best=max(rs,key=lambda v:(v['metrics']['ACC'],-v['seed']))
    assert best['run_id']==s['selected_run_id'] and best['seed']==s['selected_seed']
    assert best['metrics']==s['metrics'];match(best['metrics'],lookup[s['method'],s['dataset']])
for e in sources:
    if e.get('run_id'):match(run_map[e['run_id']]['metrics'],e)
for dataset,record in index['native_epoch_selections'].items():
    es=rows(safe(record['file']))
    for point in es:
        match(text_metrics(point['source_text']),point)
        assert int(re.search(r'Epoch\s+(\d+)',point['source_text']).group(1))==int(point['epoch'])
    best=max(es,key=lambda v:(float(v['ACC']),-int(v['epoch'])))
    assert int(best['epoch'])==record['selected_epoch'];match(best,lookup['SCMVC',dataset])
for record in index['source_hashes']:
    p=safe(record['package_file']);assert sha(p)==record['sha256']
    if record.get('git_blob_sha'):
        content=p.read_bytes();blob=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
        assert blob==record['git_blob_sha']

# Original own ablations, parameters and figure source values retain their checks.
receipt_checks=0
for r in rows(BASE/'tables/ablation_modules.csv'):
    assert r['table']=='4';match(receipt_values(receipt[r['source_id']]),r);receipt_checks+=4
for r in rows(BASE/'tables/parameter_onefactor.csv'):
    assert r['table']=='5';values=receipt_values(receipt[r['source_id']])
    for k in ('ACC','NMI'):assert abs(values[k]*100-float(r[k+'_percent']))<1e-10;receipt_checks+=1
for r in rows(BASE/'figures/fig5_fscore.csv'):
    values=receipt_values(receipt[r['source_id']]);assert values['Fscore']==float(r['Fscore']);receipt_checks+=1
for r in rows(BASE/'figures/fig7_fig8_loss_grid.csv')+rows(BASE/'figures/fig9_neighbors.csv'):
    values=receipt_values(receipt[r['source_id']])
    for k in ('ACC','NMI'):assert abs(values[k]*100-float(r[k+'_percent']))<1e-10;receipt_checks+=1
for r in rows(BASE/'tables/strategy_checks.csv'):match(receipt_values(receipt[r['source_id']]),r);receipt_checks+=4
additional=rows(BASE/'tables/additional_literature_reference.csv')
assert len(additional)==14 and all(r['fixed_main_comparison_member']=='false' for r in additional)
assert sum(r['method'].startswith('ACCMVC') for r in additional)==6
assert sum(r['method'].startswith('SLR-MVTC') for r in additional)==3
source_recovery=manifest['Mfeat_historical_source_reconstruction']
assert source_recovery['status']=='historical_six_component_bundle_exactly_reconstructed_and_Mfeat_active_path_equivalence_established'
import base64
import ast
recipe=read(safe(source_recovery['recipe_file']))
assert sha(safe(source_recovery['recipe_file']))==source_recovery['recipe_sha256']
historical={}
for component in recipe['components']:
    source=ROOT/component['package_source_file'];raw=source.read_bytes()
    assert sha(source)==component['package_source_sha256']
    lines=raw.splitlines(keepends=True);chunks=[];position=0
    for edit in component['line_byte_replacements']:
        assert position<=edit['start_line_0based']<=edit['end_line_0based']<=len(lines)
        chunks.append(b''.join(lines[position:edit['start_line_0based']]))
        chunks.append(base64.b64decode(edit['replacement_base64']))
        position=edit['end_line_0based']
    chunks.append(b''.join(lines[position:]));data=b''.join(chunks)
    assert hashlib.sha256(data).hexdigest()==component['reconstructed_sha256']
    historical[component['name']]=data
bundle=hashlib.sha256()
for name in sorted(historical):
    bundle.update(name.encode('utf-8'));bundle.update(b'\0');bundle.update(hashlib.sha256(historical[name]).digest())
target='7D53FEC852B9BEF89270A4EA30D402CE54F4EA4E7A19C2B0CAA77B04AA7B4940'
assert bundle.hexdigest().upper()==target==recipe['target_bundle_sha256']==source_recovery['target_bundle_sha256']
assert historical['model.py']==safe(source_recovery['historical_model_file']).read_bytes()
mf=receipt['sacr_Mfeat_404_Full']['result']['configuration']
assert mf['con_epochs']==0 and mf['resolved_pretraining_code_sha256']==target
print(json.dumps(dict(status='PASS',scope='stdlib records, selections and integrity; arrays not recomputed by this script',
    main_rows=66,entry_sources=66,baseline_entries=60,source_kind_counts=dict(kind_counts),
    prediction_file_hashes=37,complete_run_selections=5,SCMVC_epoch_selections=2,
    protected_base_files_unchanged=len(manifest['protected_base_files']),source_snapshot_files=len(index['source_hashes']),
    all_package_files_checksummed=len(checks),additional_reference_entries=14,own_auxiliary_numeric_checks=receipt_checks,
    historical_Mfeat_source_bundle_reconstructed=target,
    models_loaded=0,inference_or_training_or_KMeans_fit=0),indent=2))
