from pathlib import Path
import json,shutil
import joblib
from ml.dataset import sha256,write_json
from scripts.train_generalized import annotated_rows,evaluate
root=Path.cwd();reportpath=root/'artifacts/action_v8_evaluation.json';report=json.loads(reportpath.read_text(encoding='utf-8'))
for name in ['typed_oof','action_transfer_oof']:
 v=report[name]['videos'];positives=[r for r in v if r['expected']!='normal']
 report[name]['correct_event_type_videos']=sum(any(e['type']==r['expected'] for e in r['events']) for r in positives)
 report[name]['positive_videos']=len(positives)
for item in report['action_states'].values():item['evaluated_samples']=sum(map(sum,item['confusion_matrix']))
report['decision']='Reject both new violation detectors: excessive false positives and poor event typing. Release new supervised state heads with previous category classifier; do not claim new-video violation accuracy.'
write_json(reportpath,report)
for name in ['action_feedback_v8','action_transfer_v8']:
 p=root/'models'/f'{name}.json';m=json.loads(p.read_text(encoding='utf-8'));m['deployment_status']='rejected_for_violation_generalization';write_json(p,m)
old=joblib.load(root/'models/action_feedback_v7.joblib');candidate=joblib.load(root/'models/action_feedback_v8.joblib');transfer=joblib.load(root/'models/action_transfer_v8.joblib')
old.presence=candidate.presence;old.customer_presence=transfer.heads[1]
p=root/'models/action_states_v8.joblib';assert not p.exists();joblib.dump(old,p)
meta=json.loads((root/'models/action_feedback_v7.json').read_text(encoding='utf-8'))
v8=json.loads((root/'models/action_feedback_v8.json').read_text(encoding='utf-8'))
meta.update(pipeline_sha256=sha256(p),action_annotations_sha256=v8['action_annotations_sha256'],action_train_ids=v8['action_train_ids'],action_train_sha256=v8['action_train_sha256'],
 remaining_holdout_ids=[],deployment_status='state_heads_update_only',evaluation_note=report['decision'],
 classifier='v7_category_head_with_v8_presence_heads',category_training_quarantine_not_applied_to_legacy_head=True,
 limitations='Legacy categories retain their earlier weak labels. New state heads trained on 40 videos; neither state classification nor known-video regression proves unseen-video violation accuracy.')
write_json(p.with_suffix('.json'),meta)
rows=annotated_rows();write_json(root/'artifacts/action_states_v8_regression.json',evaluate(old,rows,.55))
pipeline=root/'ml/pipeline.py';s=pipeline.read_text(encoding='utf-8')
needle="    return result\n";assert s.count(needle)==1
replacement="""    if hasattr(classifier, 'customer_presence'):
        operator = classifier.presence.predict_proba(x)[:, list(classifier.presence.classes_).index(1)]
        customer = classifier.customer_presence.predict_proba(x)[:, list(classifier.customer_presence.classes_).index(1)]
        observations = []
        for w, op, cp in zip(windows, operator, customer):
            op_text = 'кассир у рабочего места' if op >= .8 else 'рабочее место предположительно пустует' if op <= .2 else 'присутствие кассира неясно'
            cp_text = 'клиент возле стола' if cp >= .8 else 'клиент возле стола не замечен' if cp <= .2 else 'присутствие клиента неясно'
            observations.append(dict(timestamp_sec=w['timestamp'], action='workstation_and_customer_state',
                detail=op_text + '; ' + cp_text + '. Гипотеза модели по четырёхсекундному фрагменту.'))
        result['observations'] = observations
    return result
"""
backup=root/'artifacts/action_states_v8_backup';backup.mkdir(exist_ok=True);shutil.copy2(pipeline,backup/'pipeline.py');shutil.copy2(root/'.env',backup/'.env')
pipeline.write_text(s.replace(needle,replacement),encoding='utf-8')
for key in ['typed_oof','action_transfer_oof']:print(key,report[key]['metrics'],'correct types',report[key]['correct_event_type_videos'])
print('Saved state update; activation pending checks')
