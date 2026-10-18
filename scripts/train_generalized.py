"""Grouped out-of-fold development selection; held-out videos evaluated once afterwards."""
import json
from pathlib import Path
import joblib,numpy as np
from scipy.special import expit
from sklearn.model_selection import StratifiedGroupKFold
from ml.generalized import features,fit_binary,GeneralizedClassifier
from ml.typed import records,fit
from ml.dataset import write_json,sha256

ROOT=Path(__file__).resolve().parents[1]
HOLDOUT={'x-10','x-16','x-27','x-46'}
FEEDBACK={'x-2','x-4','x-22','x-23','x-25','x-30'}

def annotated_rows():
    rows=records(ROOT)
    for r in rows:
        ident=r['row']['id']
        if ident in ('x-2','x-25','x-22'):
            lo,hi={'x-2':(8.,17.),'x-25':(25.,32.),'x-22':(15.5,21.1)}[ident]
            r['eligible']=np.array([w['end']>lo and w['start']<hi for w in r['windows']]);r['weak']=False
    return rows

def hard_negatives(rows):
    extra=[]
    for r in rows:
        cutoff={'x-2':5.,'x-25':24.,'x-22':12.}.get(r['row']['id'])
        if cutoff is None:continue
        mask=np.array([w['end']<=cutoff for w in r['windows']])
        if mask.any():extra.append(dict(r,label='normal',eligible=mask,weak=False))
    return rows+extra

def stats(labels,predictions):
    y=np.asarray(labels,dtype=bool);p=np.asarray(predictions,dtype=bool)
    tp=int((y&p).sum());fp=int((~y&p).sum());fn=int((y&~p).sum());tn=int((~y&~p).sum())
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,f1=2*tp/max(1,2*tp+fp+fn),precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn))

def evaluate(model,rows,threshold):
    out=[]
    for r in rows:
        events=model.events(r['x'],r['windows'],threshold)
        out.append(dict(id=r['row']['id'],expected=r['label'],positive=bool(events),events=[dict(type=e.get('event_type','unknown'),time=e['peak']) for e in events]))
    return dict(metrics=stats([r['expected']!='normal' for r in out],[r['positive'] for r in out]),videos=out)

def main(rows=None,model_name='generalized_v1',report_name='generalized_evaluation',fallback_name='typed_feedback_v6',image_mode='full_frame',sampling=None):
    target=ROOT/'models'/(model_name+'.joblib')
    if target.exists():raise ValueError('Refusing overwrite')
    rows=annotated_rows() if rows is None else rows;dev=[r for r in rows if r['row']['id'] not in HOLDOUT];held=[r for r in rows if r['row']['id'] in HOLDOUT]
    assert not {r['row']['sha256'] for r in dev}&{r['row']['sha256'] for r in held}
    y=np.array([r['label']!='normal' for r in dev])
    # Paired near-identical operations kept in one fold, not split by frame.
    pairs={'x-22':'pair22_23','x-23':'pair22_23','x-12':'pair12_24','x-24':'pair12_24'}
    groups=[pairs.get(r['row']['id'],r['row']['sha256']) for r in dev]
    folds=list(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=42).split(np.zeros(len(dev)),y,groups))
    candidates=[];best=None
    for mode in ('appearance','relative','combined'):
        for alpha in (10.,100.):
            predictions=np.zeros(len(dev))
            for train,val in folds:
                train_rows=[dev[i] for i in train];m=fit_binary(hard_negatives(train_rows),mode,alpha)
                for i in val:predictions[i]=expit(m.decision_function(features(dev[i]['x'],mode))).max()
            thresholds=sorted(set([.95]+[float(t) for t in predictions]))
            for threshold in thresholds:
                metric=stats(y,predictions>=threshold)
                # Explicit false-positive budget; same policy for all candidates.
                if metric['fp']>2:continue
                rank=(metric['f1'],-metric['fp'],metric['recall'])
                candidate=dict(mode=mode,alpha=alpha,threshold=threshold,metrics=metric)
                if best is None or rank>best[0]:best=(rank,candidate)
            candidates.append(dict(mode=mode,alpha=alpha,oof=[dict(id=r['row']['id'],expected=bool(y[i]),score=float(predictions[i])) for i,r in enumerate(dev)]))
            print('evaluated',mode,alpha,'best',best[1],flush=True)
    chosen=best[1];training=hard_negatives(dev)
    detector=fit_binary(training,chosen['mode'],chosen['alpha']);typed=fit(training)
    candidate=GeneralizedClassifier(detector,chosen['mode'],typed)
    baseline=joblib.load(ROOT/'models/typed_feedback_v5.joblib')
    original_by_id={r['row']['id']:r for r in annotated_rows()}
    baseline_held=[original_by_id[r['row']['id']] for r in held]
    report=dict(chosen=chosen,candidates=candidates,
        folds=[dict(train_ids=[dev[i]['row']['id'] for i in tr],validation_ids=[dev[i]['row']['id'] for i in va]) for tr,va in folds],
        holdout_before=evaluate(baseline,baseline_held,.55),holdout_after=evaluate(candidate,held,chosen['threshold']),
        development_after=evaluate(candidate,dev,chosen['threshold']),
        note='Model/threshold selected only on grouped OOF development predictions. Remaining holdout previously inspected in earlier work; not a fresh independent benchmark. No unseen-camera data. No holdout retuning.')
    joblib.dump(candidate,target)
    meta=json.loads((ROOT/'models/typed_feedback_v5.json').read_text())
    meta.update(pipeline_sha256=sha256(target),classifier='separate_temporal_binary_and_typed_heads',threshold=chosen['threshold'],
        image_mode=image_mode,sampling=sampling or meta['sampling'],threshold_objective='grouped_3_fold_oof_f1_with_at_most_2_false_positives',feature_mode=chosen['mode'],alpha=chosen['alpha'],
        train_ids=[r['row']['id'] for r in dev],train_sha256=[r['row']['sha256'] for r in dev],
        validation_ids=[r['row']['id'] for r in dev],validation_refit=True,remaining_holdout_ids=sorted(HOLDOUT),
        annotations_sha256=sha256(ROOT/'data/raw/team_annotations.json'),event_types=[str(c) for c in typed.estimator.classes_ if c!='normal'],evaluation_note=report['note'])
    write_json(target.with_suffix('.json'),meta);write_json(ROOT/'artifacts'/(report_name+'.json'),report)
    # Also save updated typed-only fallback. Compare before choosing a release.
    fallback=ROOT/'models'/(fallback_name+'.joblib');joblib.dump(typed,fallback)
    typed_meta=json.loads((ROOT/'models/typed_feedback_v5.json').read_text());typed_meta.update(image_mode=image_mode,sampling=sampling or typed_meta['sampling'],pipeline_sha256=sha256(fallback),annotations_sha256=meta['annotations_sha256'],train_ids=meta['train_ids'],train_sha256=meta['train_sha256'],feedback_ids=sorted(FEEDBACK),event_types=meta['event_types'])
    write_json(fallback.with_suffix('.json'),typed_meta)
    write_json(ROOT/'artifacts'/(fallback_name+'_regression.json'),dict(holdout=evaluate(typed,held,.55),development=evaluate(typed,dev,.55)))
    print(json.dumps({k:v for k,v in report.items() if k not in ('candidates','folds','development_after')},ensure_ascii=True,indent=2),flush=True)

if __name__=='__main__':main()
