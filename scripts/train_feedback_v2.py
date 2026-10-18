"""User-confirmed X-4 feedback. This video is now development, not holdout."""
import json
from pathlib import Path
import joblib
from ml.typed import records,fit
from ml.dataset import sha256,write_json

def main():
    root=Path(__file__).resolve().parents[1]
    target=root/'models/typed_feedback_v2.joblib'
    if target.exists(): raise ValueError('Refusing to overwrite feedback model')
    rows=records(root)
    training=[r for r in rows if r['row']['split']!='test' or r['row']['id']=='x-4']
    model=fit(training)
    threshold=.55
    old=joblib.load(root/'models/typed_classifier.joblib')
    details=[]
    for r in rows:
        before=old.events(r['x'],r['windows'],threshold)
        after=model.events(r['x'],r['windows'],threshold)
        details.append(dict(id=r['row']['id'],expected=r['label'],used_for_training=r in training,
            before=[dict(type=e.get('event_type','unknown'),time=e['peak']) for e in before],
            after=[dict(type=e.get('event_type','unknown'),time=e['peak']) for e in after]))
    joblib.dump(model,target)
    meta=json.loads((root/'models/typed_classifier.json').read_text())
    meta.update(pipeline_sha256=sha256(target),train_ids=[r['row']['id'] for r in training],
        train_sha256=[r['row']['sha256'] for r in training],test_used_for_training=True,
        feedback_ids=['x-4'],feedback_aliases=['x-47'],
        evaluation_note='Original holdout X-4/X-47 explicitly incorporated following user feedback; no independent test-quality claim.',
        remaining_holdout_ids=[r['row']['id'] for r in rows if r['row']['split']=='test' and r['row']['id']!='x-4'])
    write_json(target.with_suffix('.json'),meta)
    write_json(root/'artifacts/typed_feedback_v2_regression.json',dict(videos=details,note=meta['evaluation_note']))
    print(json.dumps(details,ensure_ascii=True,indent=2))

if __name__=='__main__': main()