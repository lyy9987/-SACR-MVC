"""Check table/chart CSVs against the included, retained experiment receipts.

This verifies reported numbers. It does not rerun training or independently
reproduce external methods' original-paper results.
"""
import csv
import hashlib
import json
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
R={}
for line in (BASE/'receipts/frozen_selected.jsonl').read_text(encoding='utf-8').splitlines():
    o=json.loads(line);assert o['exit_code']==0 and o['status']=='completed'
    R[o['id']]=o['result']['primary_metrics']
for p in (BASE/'receipts').glob('sacr_*.json'):
    o=json.loads(p.read_text(encoding='utf-8'));assert o['exit_code']==0
    R[o['id']]=o['metrics']
line=(BASE/'receipts/cifar10_promoted_RESULT_JSON.txt').read_text(encoding='utf-8').strip()
assert line.startswith('RESULT_JSON ')
R['cifar10_r5_svd384_tw20_seed2424']=json.loads(line[12:])['metrics']

def rows(name):
    with (BASE/name).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

checks=0
for r in rows('tables/main_comparison.csv'):
    if r['evidence_kind']!='run_receipt':continue
    for k in ('ACC','NMI','PUR','ARI'):
        assert f'{R[r["source_id"]][k]:.4f}'==f'{float(r[k]):.4f}'
        checks+=1
main={(r['method'].split(' (')[0],r['dataset']):r for r in rows('tables/main_comparison.csv')}
for line in (BASE/'receipts/baseline_selected.jsonl').read_text(encoding='utf-8').splitlines():
    obj=json.loads(line);reported=main[obj['method'],obj['dataset']]
    for k in ('ACC','NMI','PUR','ARI'):
        if reported[k]:
            assert f'{float(obj[k]):.4f}'==f'{float(reported[k]):.4f}'
            checks+=1
for r in rows('tables/ablation_modules.csv'):
    for k in ('ACC','NMI','PUR','ARI'):
        assert R[r['source_id']][k]==float(r[k]);checks+=1
for r in rows('tables/parameter_onefactor.csv'):
    m=R[r['source_id']]
    for k in ('ACC','NMI'):
        assert m[k]*100==float(r[k+'_percent']);checks+=1
    full_id=f'sacr_{r["dataset"]}_{303 if r["dataset"]=="RGB-D" else 101}_Full'
    assert abs(float(r['delta_ACC_pp'])-(m['ACC']-R[full_id]['ACC'])*100)<1e-12
    checks+=1
for r in rows('figures/fig5_fscore.csv'):
    assert R[r['source_id']]['Fscore']==float(r['Fscore'])
    assert R[r['source_id']]['Fscore']*100==float(r['Fscore_percent']);checks+=1
for r in rows('figures/fig7_fig8_loss_grid.csv')+rows('figures/fig9_neighbors.csv'):
    for k in ('ACC','NMI'):
        assert R[r['source_id']][k]*100==float(r[k+'_percent']);checks+=1
for r in rows('tables/strategy_checks.csv'):
    for k in ('ACC','NMI','PUR','ARI'):
        assert R[r['source_id']][k]==float(r[k]);checks+=1
    full=f'sacr_{r["dataset"]}_{404 if r["dataset"]=="Mfeat" else 101}_Full'
    assert abs((R[full]['ACC']-R[r['source_id']]['ACC'])*100-float(r['ACC_drop_pp']))<1e-12
    checks+=1
for line in (BASE/'CHECKSUMS.sha256').read_text(encoding='utf-8').splitlines():
    expected,name=line.split('  ',1)
    assert hashlib.sha256((BASE/name).read_bytes()).hexdigest()==expected,name
print(f'PASS: {checks} receipt-based numeric checks; all listed file checksums match.')
