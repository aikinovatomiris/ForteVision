from pathlib import Path
import json
from ml.dataset import sha256,write_json
from scripts.train_actions import BOXES_A,BOXES_B
ROOT=Path.cwd()
# Reviewed snapshot indices, not inferred dense timelines. Omitted = unknown.
STATES={1:('all',[]),10:('all',[]),11:([0,1,2,3,4,5],[9,10,11]),
13:(list(range(9)),[11]),15:(list(range(8)),[10,11]),20:('all',[]),21:('all',[]),
26:(list(range(7)),[10,11]),27:(list(range(8)),[10,11]),28:(list(range(7)),[9,10,11]),
29:(list(range(7)),[10,11]),3:('all',[]),31:(list(range(5,12)),[0,1,2]),
32:(list(range(4,12)),[0,1]),33:([],[0,1,2]),34:('all',[]),35:('all',[]),37:('all',[]),
38:('all',[]),39:('all',[]),4:('all',[]),40:(list(range(4,12)),[0]),43:('all',[]),44:('all',[]),45:('all',[]),46:('all',[]),5:('all',[]),8:('all',[])}
CUSTOMERS={1:([3,4,5,6,7,8,9,10,11],[0]),3:(list(range(3,12)),[0]),
4:([2,3,4,5,6,7,10,11],[0,8]),5:([1,2,3,4,5,6,7,8,11],[0,9]),
34:([4,5,6,7],[0,10,11]),35:(list(range(2,10)),[0,11]),37:(list(range(5,12)),[0,1]),
38:([2,3,4,7,8,9],[0,5,6,11]),39:([1,2,3,4,7,8,9,10],[0,5,6,11])}
AUDIT={'x-8':'Description says departure; not seen in reviewed snapshots.',
       'x-14':'Description says missing customer document; no customer interaction seen in reviewed snapshots.',
       'x-15':'Description says overlapping customers; reviewed snapshots show one operator and departure.',
       'x-21':'Description says departure; operator remains at workstation in reviewed snapshots.'}

dataset=json.loads((ROOT/'data/annotations/actions_objects_v1.json').read_text(encoding='utf-8'))
manifest=json.loads((ROOT/'artifacts/quick_manifest.json').read_text(encoding='utf-8'))
rows={r['id']:r for r in manifest['rows']}
for number,(present,absent) in STATES.items():
 ident=f'x-{number}';review=json.loads((ROOT/'artifacts/action_review_remaining'/f'{ident}.json').read_text())
 if present=='all':present=list(range(12))
 times=review['times'];actions=[]
 for label,indices in [(1,present),(0,absent)]:
  for i in indices:actions.append(dict(time_sec=times[i],action='cashier_at_workstation',value=bool(label),source='assistant_visual_review_sparse',boundary_precision='snapshot only'))
 cp,cn=CUSTOMERS.get(number,([],[]))
 # No visible customer at these snapshots in solo-operator recordings.
 if number not in CUSTOMERS and number!=33:cn=list(range(12))
 for label,indices in [(1,cp),(0,cn)]:
  for i in indices:actions.append(dict(time_sec=times[i],action='customer_near_counter',value=bool(label),source='assistant_visual_review_sparse',boundary_precision='snapshot only'))
 boxes=BOXES_A if number in (1,3,4,5) else BOXES_B
 if number in (34,35,37,38,39):boxes={'storage':[.72,.40,.97,.56],'laptop':[.08,.40,.28,.52],'cash_counter':[.50,.36,.69,.48]}
 if number in (31,32,33,40,43,44,45,46):boxes={'storage':[.43,.32,.64,.47],'laptop':[.10,.40,.28,.50],'cash_counter':[.49,.39,.66,.51]}
 dataset['videos'].append(dict(id=ident,sha256=rows[ident]['sha256'],duration_seconds=review['duration'],
  reviewed_snapshots=[dict(time_sec=t,reviewed=True) for t in times],
  objects=[dict(time_sec=0.,label=k,bbox_xyxy=b,source='assistant_visual_review',state='unknown') for k,b in boxes.items()],
  actions=actions,unknown=['document_identity','key_state','lock_rotation','between_snapshot_actions'],
  annotation_audit=AUDIT.get(ident),role_note='x33 later person role not established; left unlabeled' if number==33 else None))
assert len(dataset['videos'])==len(rows)==40
assert len({v['sha256'] for v in dataset['videos']})==40
dataset.update(version=2,coverage='All 40 unique existing videos sparsely reviewed; two duplicate files share original annotations. Not dense frame-by-frame labels.',
 category_training_quarantine=AUDIT,note='Original labels preserved. Quarantine is an uncertainty flag from sparse review, not a replacement normal label. No invented document or key boxes.')
write_json(ROOT/'data/annotations/actions_objects_v2.json',dataset)
print('videos',len(dataset['videos']),'snapshots',sum(len(v['reviewed_snapshots']) for v in dataset['videos']),'boxes',sum(len(v['objects']) for v in dataset['videos']))
