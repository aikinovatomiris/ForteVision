"""Video-group validation and untouched-by-fit holdout for temporal model."""
import json
from pathlib import Path
from collections import Counter
import joblib,numpy as np
from ml.typed import records
from ml.sequence import fit_sequence
from ml.dataset import write_json,sha256

def report(model,rows,threshold):
    details=[]
    for r in rows:
        events=model.events(r['x'],r['windows'],threshold)
        details.append(dict(id=r['row']['id'],expected=r['label'],positive=bool(events),
            predicted=sorted({e.get('event_type','unknown') for e in events}),
            times=[e['peak'] for e in events]))
    tp=sum(r['positive'] and r['expected']!='normal' for r in details)
    fp=sum(r['positive'] and r['expected']=='normal' for r in details)
    fn=sum(not r['positive'] and r['expected']!='normal' for r in details)
    tn=sum(not r['positive'] and r['expected']=='normal' for r in details)
    return dict(confusion_matrix=[[tn,fp],[fn,tp]],f1=2*tp/max(1,2*tp+fp+fn),videos=details)

def main():
    root=Path(__file__).resolve().parents[1];rows=records(root)
    dev=[r for r in rows if r['row']['split']!='test' or r['row']['id']=='x-4']
    holdout=[r for r in rows if r['row']['split']=='test' and r['row']['id']!='x-4']
    # One full recording from each repeated category, deterministic, no holdout inspection.
    count=Counter(r['label'] for r in dev);used=Counter();train=[];val=[]
    for r in sorted(dev,key=lambda r:r['row']['id']):
        limit=3 if r['label']=='normal' else 1
        if count[r['label']]>=2 and used[r['label']]<limit:
            val.append(r);used[r['label']]+=1
        else:train.append(r)
    assert not {r['row']['sha256'] for r in train}&{r['row']['sha256'] for r in val}
    assert not {r['row']['sha256'] for r in dev}&{r['row']['sha256'] for r in holdout}
    candidates=[];best=None
    for c in (.003,.03):
        print('fit candidate',c,'train',len(train),'validation',len(val),flush=True)
        m=fit_sequence(train,c)
        for threshold in (.45,.55,.65):
            result=report(m,val,threshold)
            tn,fp=result['confusion_matrix'][0];fn,tp=result['confusion_matrix'][1]
            rank=(result['f1'],-fp)
            candidates.append(dict(c=c,threshold=threshold,report=result))
            if best is None or rank>best[0]:best=(rank,c,threshold)
    _,c,threshold=best
    print('final fit',c,threshold,flush=True)
    model=fit_sequence(dev,c)
    path=root/'models/sequence_v1.joblib'
    if path.exists():raise ValueError('Refusing overwrite')
    joblib.dump(model,path)
    meta=json.loads((root/'models/typed_feedback_v2.json').read_text())
    meta.update(pipeline_sha256=sha256(path),classifier='ordered_temporal_logistic',threshold=threshold,
        threshold_objective='development_video_f1_then_fewer_false_positives',temporal_context_seconds=12,
        feature_description='appearance, video-relative appearance, intra-window variation, signed past/future differences',
        validation_ids=[r['row']['id'] for r in val],validation_refit=True,
        train_ids=[r['row']['id'] for r in dev],train_sha256=[r['row']['sha256'] for r in dev],
        sequence_capability='Temporal class prediction; atomic action detectors and rule verification are not available',
        limitations='Small dataset; no unseen-camera benchmark; weak event intervals; hypotheses only.')
    write_json(path.with_suffix('.json'),meta)
    baseline=joblib.load(root/'models/typed_feedback_v2.joblib')
    result=dict(candidates=candidates,chosen=dict(c=c,threshold=threshold),
        baseline_holdout=report(baseline,holdout,.55),sequence_holdout=report(model,holdout,threshold),
        development_regression=report(model,dev,threshold),
        limitations='Remaining previously inspected holdout, not a fresh test. Entire content groups excluded from fitting. X-4 is development after user feedback. No camera diversity test.')
    write_json(root/'artifacts/sequence_evaluation.json',result)
    print(json.dumps(dict(chosen=result['chosen'],baseline=result['baseline_holdout'],sequence=result['sequence_holdout']),ensure_ascii=True,indent=2))
if __name__=='__main__':main()