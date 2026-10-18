"""Explicit team feedback; remaining whole videos excluded from training."""
import json
from pathlib import Path
import joblib
import numpy as np
from ml.typed import records,fit
from ml.dataset import sha256,write_json

def main():
    root=Path(__file__).resolve().parents[1]
    target=root/'models/typed_feedback_v4.joblib'
    if target.exists(): raise ValueError('Refusing to overwrite v3 model')
    rows=records(root)
    feedback={'x-4','x-2','x-25'}
    for r in rows:
        ident=r['row']['id']
        if ident in ('x-2','x-25'):
            # Approximate event context from team confirmation and local frame review.
            # X-2: arrival then cash operation; X-25: cashier exits after storage use.
            lo,hi=(8.,17.) if ident=='x-2' else (25.,32.)
            r['eligible']=np.array([w['end']>lo and w['start']<hi for w in r['windows']])
            r['weak']=False
    training=[r for r in rows if r['row']['split']!='test' or r['row']['id'] in feedback]
    holdout=[r for r in rows if r['row']['split']=='test' and r['row']['id'] not in feedback]
    hard_negatives=[]
    for r in training:
        ident=r['row']['id']
        if ident not in ('x-2','x-25'): continue
        cutoff=5. if ident=='x-2' else 24.
        mask=np.array([w['end']<=cutoff for w in r['windows']])
        if mask.any(): hard_negatives.append(dict(r,label='normal',eligible=mask,weak=False))
    model=fit(training+hard_negatives); threshold=.55
    old=joblib.load(root/'models/typed_feedback_v2.joblib')
    train_ids={r['row']['id'] for r in training}
    details=[]
    for r in rows:
        before=old.events(r['x'],r['windows'],threshold)
        after=model.events(r['x'],r['windows'],threshold)
        details.append(dict(id=r['row']['id'],expected=r['label'],used_for_training=r['row']['id'] in train_ids,
            before=[dict(type=e.get('event_type','unknown'),time=e['peak']) for e in before],
            after=[dict(type=e.get('event_type','unknown'),time=e['peak']) for e in after]))
    def metrics(items,key):
        tp=sum(bool(r[key]) and r['expected']!='normal' for r in items)
        fp=sum(bool(r[key]) and r['expected']=='normal' for r in items)
        fn=sum(not r[key] and r['expected']!='normal' for r in items)
        tn=sum(not r[key] and r['expected']=='normal' for r in items)
        return dict(videos=len(items),tp=tp,fp=fp,fn=fn,tn=tn,f1=2*tp/max(1,2*tp+fp+fn),
            correct_type=sum(any(e['type']==r['expected'] for e in r[key]) for r in items if r['expected']!='normal'))
    held=[r for r in details if not r['used_for_training']]
    negatives=[r for r in details if r['expected']=='normal']
    report=dict(videos=details,holdout_before=metrics(held,'before'),holdout_after=metrics(held,'after'),
        normal_regression_before=metrics(negatives,'before'),normal_regression_after=metrics(negatives,'after'),
        note='X-2, X-25 and earlier X-4/X-47 feedback are training examples, NOT an independent benchmark. Remaining holdout is small and previously inspected; no unseen-camera claim.')
    joblib.dump(model,target)
    meta=json.loads((root/'models/typed_feedback_v2.json').read_text())
    meta.update(pipeline_sha256=sha256(target),train_ids=[r['row']['id'] for r in training],
        train_sha256=[r['row']['sha256'] for r in training],test_used_for_training=True,
        feedback_ids=sorted(feedback),feedback_aliases=['x-47'],evaluation_note=report['note'],
        remaining_holdout_ids=[r['row']['id'] for r in holdout],
        limitations='Experimental weak supervision; known-case feedback included. No independent different-camera validation. Scores are uncalibrated.',
        verified_normal_intervals={'x-2': {'end_lte': 5.}, 'x-25': {'end_lte': 24.}},
        feedback_annotation=dict(x2='User confirmed no document handover; approximate context 8-17 s',x25='User confirmed unsecured safe; local frames show open storage after departure, context 25-32 s'),
        event_types=[str(c) for c in model.estimator.classes_ if c!='normal'])
    write_json(target.with_suffix('.json'),meta)
    write_json(root/'artifacts/typed_feedback_v4_regression.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='videos'},ensure_ascii=True,indent=2))
    print(json.dumps([r for r in details if r['id'] in feedback or not r['used_for_training']],ensure_ascii=True,indent=2))

if __name__=='__main__':main()
