from pathlib import Path
import json
import numpy as np
from ml.typed import fit
from ml.triage import route_events,add_review_candidates
from ml.dataset import write_json
from scripts.train_generalized import annotated_rows,hard_negatives,stats
ROOT=Path(__file__).resolve().parents[1]
rows={r['row']['id']:r for r in annotated_rows()}
for ident,r in rows.items():
 if ident=='x-17':r['eligible']=np.array([w['end']>7 for w in r['windows']]);r['weak']=False
folds=json.loads((ROOT/'artifacts/action_v8_evaluation.json').read_text())['folds']
reports=[]
for number,f in enumerate(folds):
 training=[rows[i] for i in f['train_ids']];extra=[]
 for r in training:
  if r['row']['id']=='x-17':extra.append(dict(r,label='normal',eligible=np.array([w['end']<=5 for w in r['windows']]),weak=False))
 model=fit(hard_negatives(training)+extra)
 for ident in f['validation_ids']:
  r=rows[ident];old=model.events(r['x'],r['windows'],.55);sensitive=model.events(r['x'],r['windows'],.35)
  combined=add_review_candidates(old,sensitive);alerts,review=route_events(combined)
  # Sensitive-only additions are never promoted to automatic alerts.
  review+= [e for e in alerts if e.get('review_only')];alerts=[e for e in alerts if not e.get('review_only')]
  reports.append(dict(id=ident,expected=r['label'],before=bool(old),alert=bool(alerts),review=bool(review),
   old_count=len(old),new_count=len(combined),correct_alert_type=any(e.get('event_type')==r['label'] for e in alerts),
   alerts=[dict(type=e.get('event_type','unknown'),time=e['peak'],score=e['score']) for e in alerts],
   review_intervals=[dict(type=e.get('event_type','unknown'),start=e['start'],end=e['end']) for e in review]))
 print('fold',number+1,'complete',flush=True)
y=[r['expected']!='normal' for r in reports]
report=dict(thresholds=dict(alert=.8,review=.35,previous=.55),thresholds_tuned=False,
 before=stats(y,[r['before'] for r in reports]),alerts=stats(y,[r['alert'] for r in reports]),
 alerts_or_review=stats(y,[r['alert'] or r['review'] for r in reports]),
 correct_alert_types=sum(r['correct_alert_type'] for r in reports if r['expected']!='normal'),
 candidates_preserved=all(r['new_count']>=r['old_count'] for r in reports),videos=reports,
 note='Routing and sensitivity change, not proof of better visual recognition. Coverage is video-level, not exact moment recall. Review workload is reported as positives on normal videos. Same previously inspected dataset, grouped refits.')
write_json(ROOT/'artifacts/triage_evaluation.json',report)
print(json.dumps({k:v for k,v in report.items() if k!='videos'}),flush=True)
