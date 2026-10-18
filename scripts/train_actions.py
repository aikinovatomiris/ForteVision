"""Reproducible sparse manual annotations and action/object transfer learning.
Reviewed source: artifacts/action_review/*.jpg, sixteen real-PTS frames per clip.
Unknown objects/actions are never converted to negative training labels.
"""
from pathlib import Path
import json, shutil
import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from ml.dataset import sha256, write_json
from ml.preprocessing import probe, read_at_times
from ml.detection import ResNetExtractor
from ml.typed import fit
from ml.action_model import ActionAwareClassifier
from scripts.train_generalized import annotated_rows, hard_negatives, evaluate

ROOT = Path(__file__).resolve().parents[1]
# Intervals conservatively inside reviewed stationary states, not exact boundaries.
PRESENCE = {
 'x-2': [(0,17,1)], 'x-14': [(0,142,1)], 'x-16': [(0,22.7,1)],
 'x-17': [(0,14.7,1)], 'x-18': [(0,22.5,1)], 'x-19': [(0,57.3,1)],
 'x-12': [(0,18,1),(39,42.3,0)],
 'x-22': [(0,13,1),(18.2,21,0)],
 'x-23': [(0,11,1),(16,18.2,0)],
 'x-24': [(0,6.5,1),(12,14.6,0)],
 'x-25': [(0,23.5,1),(29,31.6,0)],
 'x-30': [(0,2,0),(6,18.7,1)],
}
# Approximate object boxes on explicitly reviewed frame 0 only; XYXY normalized.
# These are training annotations, never fixed inference regions.
BOXES_A = {'storage': [.68,.37,.94,.56], 'laptop':[.48,.34,.68,.48],
           'cash_counter':[.07,.39,.23,.51]}
BOXES_B = {'storage': [.43,.30,.63,.45], 'laptop':[.51,.40,.68,.49],
           'cash_counter':[.09,.44,.29,.56]}

def estimator():
    return make_pipeline(StandardScaler(), LogisticRegression(C=.01, max_iter=2000, class_weight='balanced'))

def weighted_fit(x,y,groups):
    groups=np.asarray(groups);weights=np.array([1/sum(groups==g) for g in groups]);weights*=len(weights)/weights.sum()
    return estimator().fit(np.asarray(x),np.asarray(y),logisticregression__sample_weight=weights)

def main():
    target=ROOT/'models/action_feedback_v7.joblib'
    if target.exists():raise ValueError('Refusing overwrite')
    backup=ROOT/'artifacts/feedback_16_17_backup';backup.mkdir(exist_ok=True)
    annotation_path=ROOT/'data/raw/team_annotations.json'
    shutil.copy2(annotation_path,backup/annotation_path.name)
    ann=json.loads(annotation_path.read_text(encoding='utf-8'))
    ann['events']['x-17']=['document_handover_not_observed','Клиент передал деньги и ушёл без предъявления удостоверения; подтверждено пользователем. Метка хранилища отозвана.']
    ann['corrections'].append(dict(date='2026-10-18',source='Explicit user confirmation',normal=['x-16'],events={'x-17':'document_handover_not_observed'},retracted={'x-17':'storage_left_open'}))
    write_json(annotation_path,ann)
    paths={p.stem.lower().replace('х','x'):p for p in (ROOT/'data/videos').glob('*.mov')}
    videos=[]
    for ident, intervals in PRESENCE.items():
        path=paths[ident];info=probe(path)
        snapshots=[dict(time_sec=round(i*(info['duration_seconds']-.15)/15,4),reviewed=True) for i in range(16)]
        boxes=BOXES_A if ident in ('x-2','x-16','x-17','x-18','x-19') else BOXES_B
        if ident=='x-30':boxes={'storage':[.42,.34,.61,.46], 'cash_counter':[.39,.40,.57,.50]}
        objects=[dict(time_sec=0.,label=k,bbox_xyxy=v,source='assistant_visual_review',state='unknown') for k,v in boxes.items()]
        actions=[dict(start_sec=a,end_sec=b,action='cashier_at_workstation',value=bool(c),source='assistant_visual_review_sparse',boundary_precision='approximate; between sampled frames') for a,b,c in intervals]
        if ident=='x-17':
            actions += [dict(start_sec=3.9,end_sec=6.8,action='customer_approaches',value=True,source='assistant_visual_review_sparse'),
                        dict(start_sec=6.8,end_sec=8.73,action='counter_handover',value=True,source='assistant_visual_review_sparse',object_identity='cash confirmed by user'),
                        dict(start_sec=9.73,end_sec=13.6,action='customer_leaves',value=True,source='assistant_visual_review_sparse'),
                        dict(start_sec=0,end_sec=info['duration_seconds'],action='document_presented',value=False,source='user_confirmation',boundary_precision='video_level; not framewise proof')]
        videos.append(dict(id=ident,sha256=sha256(path),duration_seconds=info['duration_seconds'],reviewed_snapshots=snapshots,objects=objects,actions=actions,unknown=['identity_document_bbox','key_bbox','lock_rotation','exact_transition_boundaries']))
    dataset=dict(schema_version=1,annotation_kind='sparse reviewed snapshots and conservative action intervals',
                 not_exhaustive=True,object_coordinate_system='normalized_xyxy',videos=videos,
                 note='Video-level normal does not label every action. Object category is not object state. No document/key boxes invented from user video labels.')
    dest=ROOT/'data/annotations/actions_objects_v1.json';dest.parent.mkdir(exist_ok=True);write_json(dest,dataset)
    rows=annotated_rows()
    for r in rows:
        if r['row']['id']=='x-17':
            r['eligible']=np.array([w['end']>7 and w['start']<14.7 for w in r['windows']]);r['weak']=False
    # X16 is now feedback/training. Never report it as holdout again.
    held_ids={'x-10','x-27','x-46'}
    train=[r for r in rows if r['row']['id'] not in held_ids];held=[r for r in rows if r['row']['id'] in held_ids]
    x=[];y=[];groups=[];ids=[]
    pair={'x-22':'pair22_23','x-23':'pair22_23','x-12':'pair12_24','x-24':'pair12_24',
          'x-16':'pair16_17','x-17':'pair16_17'}
    for r in train:
        ident=r['row']['id']
        for w,vec in zip(r['windows'],r['x']):
            for start,end,label in PRESENCE.get(ident,[]):
                if w['start']>=start and w['end']<=end+.15:
                    x.append(vec);y.append(label);groups.append(pair.get(ident,ident));ids.append(ident);break
    x=np.asarray(x);y=np.asarray(y);groups=np.asarray(groups)
    predictions=np.zeros(len(y),dtype=int);folds=[]
    for group in sorted(set(groups)):
        tr=groups!=group;va=~tr
        if len(set(y[tr]))<2:raise ValueError('Missing state in training fold')
        head=weighted_fit(x[tr],y[tr],groups[tr]);predictions[va]=head.predict(x[va])
        folds.append(dict(validation_group=group,train_ids=sorted(set(np.array(ids)[tr])),validation_ids=sorted(set(np.array(ids)[va]))))
    presence=weighted_fit(x,y,groups)
    extra=[]
    for r in train:
        if r['row']['id']=='x-17':
            mask=np.array([w['end']<=5 for w in r['windows']]);extra.append(dict(r,label='normal',eligible=mask,weak=False))
    typed=fit(hard_negatives(train)+extra);model=ActionAwareClassifier(typed,presence)
    joblib.dump(model,target)
    meta=json.loads((ROOT/'models/typed_feedback_v6.json').read_text(encoding='utf-8'))
    meta.update(pipeline_sha256=sha256(target),classifier='typed_plus_supervised_workstation_presence',
        annotations_sha256=sha256(annotation_path),action_annotations_sha256=sha256(dest),
        train_ids=[r['row']['id'] for r in train],train_sha256=[r['row']['sha256'] for r in train],
        remaining_holdout_ids=sorted(held_ids),feedback_ids=sorted(set(meta['feedback_ids'])|{'x-16','x-17'}),
        weak_positive_ids=[r['row']['id'] for r in train if r['weak']],
        evaluation_note='X16 and X17 are corrected training examples. Action validation leaves whole paired-video groups out. Same room, previously viewed data; no independent-camera claim.',
        limitations='Sparse action annotation. No trained identity-document detector. Object crop head is experimental and not a whole-frame detector. Category probabilities uncalibrated.')
    meta['verified_normal_intervals']['x-17']={'end_lte':5.}
    write_json(target.with_suffix('.json'),meta)
    report=dict(action_samples=len(y),action_confusion_matrix_labels=[0,1],action_confusion_matrix=confusion_matrix(y,predictions,labels=[0,1]).tolist(),
        action_balanced_accuracy=float(balanced_accuracy_score(y,predictions)),action_folds=folds,
        regression=evaluate(model,train,.55),remaining_holdout=evaluate(model,held,.55),
        annotation_videos=len(videos),reviewed_snapshots=sum(len(v['reviewed_snapshots']) for v in videos),
        object_boxes=sum(len(v['objects']) for v in videos),note=meta['evaluation_note'])
    write_json(ROOT/'artifacts/action_v7_evaluation.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('action_folds','regression')},ensure_ascii=True),flush=True)
    # Train object categories on annotated crops; preserve a separate honest
    # leave-video-group-out result. Not a detector: no bbox inference claimed.
    extractor=ResNetExtractor(ROOT/'models/resnet18-f37072fd.pth',image_mode='full_frame')
    crops=[];labels=[];ogroups=[]
    for v in videos:
        _,frame=read_at_times(paths[v['id']],[0])[0];h,w=frame.shape[:2]
        for obj in v['objects']:
            a,b,c,d=obj['bbox_xyxy'];crop=frame[int(b*h):int(d*h),int(a*w):int(c*w)]
            crops.append(extractor.preprocess(crop));labels.append(obj['label']);ogroups.append(pair.get(v['id'],v['id']))
    vectors=extractor.embed(crops);labels=np.asarray(labels);ogroups=np.asarray(ogroups);opred=np.empty(len(labels),dtype=object)
    for g in sorted(set(ogroups)):
        tr=ogroups!=g;opred[~tr]=weighted_fit(vectors[tr],labels[tr],ogroups[tr]).predict(vectors[~tr])
    object_head=weighted_fit(vectors,labels,ogroups);opath=ROOT/'models/object_crops_v1.joblib';joblib.dump(object_head,opath)
    write_json(opath.with_suffix('.json'),dict(pipeline_sha256=sha256(opath),annotations_sha256=sha256(dest),weights_sha256=extractor.weights_sha256,
        classes=list(object_head.classes_),samples=len(labels),accuracy=float(np.mean(labels==opred)),
        evaluation='Leave paired-video groups out, crop classification only. No object localization performance measured.',deployment_status='experimental_crop_classifier_not_detector'))
    print('Object crop grouped accuracy',float(np.mean(labels==opred)),flush=True)

if __name__=='__main__':main()
