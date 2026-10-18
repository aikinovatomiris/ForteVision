from pathlib import Path
import json
import joblib
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from ml.dataset import write_json,sha256
from ml.typed import fit
from ml.action_model import ActionAwareClassifier
from ml.action_transfer import ActionTransferClassifier,state_features
from ml.preprocessing import read_at_times,SamplingConfig
from ml.detection import ResNetExtractor
from scripts.train_actions import weighted_fit
from scripts.train_generalized import annotated_rows,hard_negatives,evaluate,stats

ROOT=Path(__file__).resolve().parents[1]
# Near-identical recordings kept together, including duplicates through content SHA.
PAIR={}
for group in [(1,2,3),(4,5),(8,10),(11,12),(13,14),(16,17,18,19),(20,21),(22,23),(24,25),(26,27),(28,29),(30,31,32),(34,35,37),(38,39),(43,44,45,46)]:
 for n in group:PAIR[f'x-{n}']='group_'+'_'.join(map(str,group))
def group(ident):return PAIR.get(ident,ident)

def main():
 target=ROOT/'models/action_feedback_v8.joblib'
 if target.exists():raise ValueError('Refusing overwrite')
 annpath=ROOT/'data/annotations/actions_objects_v2.json';ann=json.loads(annpath.read_text(encoding='utf-8'))
 videos={v['id']:v for v in ann['videos']};manifest=json.loads((ROOT/'artifacts/quick_manifest.json').read_text(encoding='utf-8'))
 paths={p.stem.lower().replace('х','x'):p for p in (ROOT/'data/videos').glob('*.mov')}
 # Reuse verified frozen backbone features, including unlabeled videos for action supervision only.
 cache={};extractor=None;baseline=json.loads((ROOT/'models/fullframe_classifier.json').read_text())
 for r in manifest['rows']:
  p=ROOT/'artifacts/quick_cache_full_frame'/f"{r['sha256']}.joblib"
  if not p.exists():
   if extractor is None:extractor=ResNetExtractor(ROOT/'models/resnet18-f37072fd.pth',image_mode='full_frame')
   x,w,_=extractor.extract(paths[r['id']],SamplingConfig(**baseline['sampling']))
   joblib.dump(dict(x=x,windows=w,sampling=baseline['sampling'],weights_sha256=extractor.weights_sha256),p)
  d=joblib.load(p)
  assert d['weights_sha256']==baseline['weights_sha256'] and d['sampling']==baseline['sampling']
  cache[r['id']]=d
 samples={name:[] for name in ['cashier_at_workstation','customer_near_counter']}
 for ident,v in videos.items():
  d=cache[ident]
  for action in v['actions']:
   if action['action'] not in samples:continue
   for i,w in enumerate(d['windows']):
    if 'time_sec' in action:
     # Label only the closest window centre; no dense propagation from sparse snapshots.
     nearest=min(range(len(d['windows'])),key=lambda j:abs(d['windows'][j]['timestamp']-action['time_sec']))
     ok=i==nearest and abs(w['timestamp']-action['time_sec'])<=1.1
    else:ok=w['start']>=action['start_sec'] and w['end']<=action['end_sec']+.15
    if ok:samples[action['action']].append((ident,i,int(action['value'])))
 # Conflicting snapshots mapped to one temporal window are unknown, not majority-voted.
 for name,items in samples.items():
  labels={}
  for ident,i,y in items:labels.setdefault((ident,i),set()).add(y)
  samples[name]=[(ident,i,next(iter(ys))) for (ident,i),ys in labels.items() if len(ys)==1]
 def heads_for(allowed):
  heads=[]
  for name,items in samples.items():
   chosen=[t for t in items if t[0] in allowed]
   heads.append(weighted_fit([cache[v]['x'][i] for v,i,y in chosen],[y for v,i,y in chosen],[group(v) for v,i,y in chosen]))
  return heads
 def negatives(rows):
  extra=[]
  for r in rows:
   if r['row']['id']=='x-17':
    mask=np.array([w['end']<=5 for w in r['windows']]);extra.append(dict(r,label='normal',eligible=mask,weak=False))
  return hard_negatives(rows)+extra
 rows=annotated_rows()
 for r in rows:
  if r['row']['id']=='x-17':
   r['eligible']=np.array([w['end']>7 for w in r['windows']]);r['weak']=False
 rows=[r for r in rows if r['row']['id'] not in ann['category_training_quarantine']]
 def fit_detector(training,heads):
  x=[];y=[];weights=[]
  for r in negatives(training):
   idx=np.flatnonzero(r['eligible']);z=state_features(r['x'],heads)
   x.extend(z[idx]);y.extend([int(r['label']!='normal')]*len(idx));weights.extend([1/len(idx)]*len(idx))
  m=make_pipeline(StandardScaler(),RidgeClassifier(alpha=10.,class_weight='balanced'))
  weights=np.asarray(weights);weights*=len(weights)/weights.sum()
  return m.fit(x,y,ridgeclassifier__sample_weight=weights)
 groups=[group(r['row']['id']) for r in rows];folds=[];typed_results=[];transfer_results=[];state_truth=[[],[]];state_pred=[[],[]]
 # 4 folds fixed before looking at predictions; every video is validated out of fold.
 for tr,va in GroupKFold(n_splits=4).split(rows,groups=groups):
  training=[rows[i] for i in tr];validation=[rows[i] for i in va]
  excluded={group(r['row']['id']) for r in validation}
  allowed={ident for ident in videos if group(ident) not in excluded}
  assert not {r['row']['sha256'] for r in training}&{r['row']['sha256'] for r in validation}
  heads=heads_for(allowed);typed=fit(negatives(training));detector=fit_detector(training,heads)
  transfer=ActionTransferClassifier(detector,heads,typed)
  for j,(name,items) in enumerate(samples.items()):
   chosen=[t for t in items if group(t[0]) in excluded]
   if chosen:
    state_truth[j].extend([y for v,i,y in chosen]);state_pred[j].extend(heads[j].predict([cache[v]['x'][i] for v,i,y in chosen]).tolist())
  typed_results.extend(evaluate(typed,validation,.55)['videos']);transfer_results.extend(evaluate(transfer,validation,.55)['videos'])
  folds.append(dict(train_ids=[r['row']['id'] for r in training],validation_ids=[r['row']['id'] for r in validation],action_train_ids=sorted(allowed)))
  print('fold',len(folds),'done',flush=True)
 def metric(v):return stats([r['expected']!='normal' for r in v],[r['positive'] for r in v])
 heads=heads_for(set(videos));typed=fit(negatives(rows));model=ActionAwareClassifier(typed,heads[0]);joblib.dump(model,target)
 transfer=ActionTransferClassifier(fit_detector(rows,heads),heads,typed);transferpath=ROOT/'models/action_transfer_v8.joblib';joblib.dump(transfer,transferpath)
 meta=json.loads((ROOT/'models/action_feedback_v7.json').read_text(encoding='utf-8'))
 meta.update(pipeline_sha256=sha256(target),action_annotations_sha256=sha256(annpath),train_ids=[r['row']['id'] for r in rows],train_sha256=[r['row']['sha256'] for r in rows],
  action_train_ids=sorted(videos),action_train_sha256=sorted(v['sha256'] for v in videos.values()),remaining_holdout_ids=[],validation_ids=[r['row']['id'] for r in rows],validation_refit=True,
  quarantined_category_ids=sorted(ann['category_training_quarantine']),event_types=[str(c) for c in typed.estimator.classes_ if c!='normal'],
  weak_positive_ids=[r['row']['id'] for r in rows if r['weak']],deployment_status='candidate_pending_regression',
  evaluation_note='4-fold grouped out-of-fold evaluation, including action heads. Previously inspected dataset; no fresh independent videos. Old X10/X27/X46 holdout now used for training and must not be claimed as held out.')
 write_json(target.with_suffix('.json'),meta)
 other=dict(meta,pipeline_sha256=sha256(transferpath),classifier='action_state_temporal_bottleneck',deployment_status='experimental_not_active');write_json(transferpath.with_suffix('.json'),other)
 report=dict(folds=folds,threshold=.55,threshold_tuned=False,typed_oof=dict(metrics=metric(typed_results),videos=typed_results),
  action_transfer_oof=dict(metrics=metric(transfer_results),videos=transfer_results),
  action_states={name:dict(samples=len(items),confusion_matrix=confusion_matrix(state_truth[j],state_pred[j],labels=[0,1]).tolist()) for j,(name,items) in enumerate(samples.items())},
  regression=evaluate(model,rows,.55),quarantined=ann['category_training_quarantine'],note=meta['evaluation_note'])
 write_json(ROOT/'artifacts/action_v8_evaluation.json',report)
 print(json.dumps({k:v for k,v in report.items() if k not in ['folds','regression','typed_oof','action_transfer_oof']}),flush=True)
 print('TYPED',metric(typed_results),'TRANSFER',metric(transfer_results),flush=True)
 # Object crop classifier; never call this an object detector.
 if extractor is None:extractor=ResNetExtractor(ROOT/'models/resnet18-f37072fd.pth',image_mode='full_frame')
 xs=[];ys=[];gs=[]
 for ident,v in videos.items():
  _,frame=read_at_times(paths[ident],[0])[0];h,w=frame.shape[:2]
  tensors=[]
  for o in v['objects']:
   a,b,c,d=o['bbox_xyxy'];tensors.append(extractor.preprocess(frame[int(b*h):int(d*h),int(a*w):int(c*w)]));ys.append(o['label']);gs.append(group(ident))
  xs.extend(extractor.embed(tensors))
 xs=np.asarray(xs);ys=np.asarray(ys);gs=np.asarray(gs);pred=np.empty(len(ys),dtype=object)
 for tr,va in GroupKFold(n_splits=4).split(xs,ys,gs):pred[va]=weighted_fit(xs[tr],ys[tr],gs[tr]).predict(xs[va])
 head=weighted_fit(xs,ys,gs);op=ROOT/'models/object_crops_v2.joblib';joblib.dump(head,op)
 write_json(op.with_suffix('.json'),dict(pipeline_sha256=sha256(op),annotations_sha256=sha256(annpath),samples=len(ys),accuracy=float(np.mean(pred==ys)),
  deployment_status='experimental_crop_classifier_not_detector',evaluation='4-fold paired-video-group OOF; same room'))
 print('objects',len(ys),float(np.mean(pred==ys)),flush=True)

if __name__=='__main__':main()
