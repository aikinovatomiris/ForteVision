"""Rebuild from the reviewed workbook, group OOF selection, frozen test evaluation.

Usage: python -m scripts.train_anchored --workbook data/raw/<file>.xlsx
Descriptions are training supervision only; neither IDs nor text enter inference.
"""
import argparse
import json
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold
from ml.anchored import fit, CATALOG
from ml.dataset import key, sha256, validate_manifest, write_json
from ml.recovery_training import read_descriptions, extract
from ml.detection import ResNetExtractor
from ml.preprocessing import SamplingConfig


def group(ident, groups):
    return next(('|'.join(sorted(g)) for g in groups if ident in g), ident)


def build_records(manifest, workbook_rows, annotations):
    records = []
    for row in manifest['rows']:
        ids = [row['id']]+[key(n) for n in row.get('aliases', [])]
        source = [workbook_rows[i] for i in ids if i in workbook_rows]
        labels = {s['binary_label'] for s in source if s['binary_label'] is not None}
        if len(labels) != 1:
            raise ValueError(f'Missing/conflicting duplicate labels: {ids}')
        label = labels.pop()
        events = annotations['events'].get(row['id'], [])
        if label not in (0,1):
            raise ValueError('Binary label must be 0 or 1')
        if events and label != 1:
            raise ValueError(f'Event contradicts normal label: {ids}')
        for event in events:
            if event['type'] not in CATALOG or not np.isfinite(event['time']) or event['time']<0:
                raise ValueError(f'Invalid timestamp/type: {ids}')
        records.append(dict(row=row, binary_label=label, events=events,
            label=None, timestamps=[e['time'] for e in events],
            timestamp_source='latest_workbook_manual_transcription', annotations=source,
            group=group(row['id'], annotations['recording_groups'])))
    return records


def measure(records, predictions, threshold):
    tp=fp=tn=fn=hits=total=typed_hits=predicted_count=0
    errors=[]
    details=[]
    for r, events in zip(records, predictions):
        detected=bool(events)
        predicted_count += len(events)
        label=r['binary_label']
        tp += int(label == 1 and detected); fn += int(label == 1 and not detected)
        fp += int(label == 0 and detected); tn += int(label == 0 and not detected)
        # One-to-one nearest temporal assignment, independent of expected type.
        pairs=sorted((abs(e['peak']-a['time']), i, j)
            for i,a in enumerate(r['events']) for j,e in enumerate(events))
        used_a=set();used_e=set()
        for distance,i,j in pairs:
            if i in used_a or j in used_e or distance > 5: continue
            used_a.add(i);used_e.add(j);hits+=1;errors.append(distance)
            typed_hits += int(events[j].get('event_type') == r['events'][i]['type'])
        total += len(r['events'])
        details.append(dict(id=r['row']['id'],label=label,expected=r['events'],events=events))
    return dict(tp=tp,fp=fp,tn=tn,fn=fn,precision=tp/(tp+fp) if tp+fp else None,
        recall=tp/(tp+fn) if tp+fn else None,f1=2*tp/max(1,2*tp+fp+fn),
        annotated_events=total,matched_within_5_seconds=hits,correct_type_and_time=typed_hits,
        predicted_events=predicted_count,
        matched_mae_seconds=float(np.mean(errors)) if errors else None,
        threshold=threshold,videos=details)


def run(args):
    root=Path(__file__).resolve().parents[1]
    out=root/'artifacts'/args.name
    target=root/'models'/f'{args.name}.joblib'
    if target.exists(): raise ValueError('Refusing to overwrite a trained model')
    workbook=Path(args.workbook)
    annotations=json.loads(Path(args.annotations).read_text(encoding='utf8'))
    if sha256(workbook)!=annotations['workbook_sha256']:
        raise ValueError('Workbook changed: review transcription before training')
    manifest=json.loads((root/'artifacts/manifest.json').read_text())
    validate_manifest(manifest,root/'data/videos')
    rows=build_records(manifest,read_descriptions(workbook),annotations)
    test_groups={r['group'] for r in rows if r['row']['split']=='test'}
    development=[r for r in rows if r['group'] not in test_groups]
    test=[r for r in rows if r['row']['split']=='test']
    quarantined=[r['row']['id'] for r in rows if r['group'] in test_groups and r['row']['split']!='test']
    fitting=[r for r in development if r['binary_label']==0 or r['events']]
    write_json(out/'data_audit.json',dict(workbook_sha256=sha256(workbook),records=rows,
        quarantined=quarantined,unlocalized_positive_ids=[r['row']['id'] for r in development if r['binary_label'] and not r['events']],
        disagreements_with_original=[r['row']['id'] for r in rows if r['binary_label']!=r['row']['label']]))
    extractor=ResNetExtractor(root/'models/resnet18-f37072fd.pth','cpu','full_frame')
    extractor.torch.set_num_threads(4)
    sampling=SamplingConfig(fps=2,window_seconds=4,stride_seconds=2,batch_size=16)
    cache=root/'artifacts/recovery_20261018'
    development=extract(development,root,cache,extractor,sampling)
    groups=[r['group'] for r in development]
    splits=list(GroupKFold(n_splits=3).split(development,groups=groups))
    thresholds=[.35,.5,.65,.8,.9] if args.objective=='f1' else [.5,.65,.8,.9,.95,.98]
    candidates=[]
    modes=['appearance','sequence'] if args.objective=='f1' else ['relative','sequence']
    for mode in modes:
        for c,weak in [(c,w) for c in [.001,.01,.1] for w in ([0.] if args.objective=='f1' else [0.,.15])]:
            print(f'cross-validation {mode} C={c} weak={weak}',flush=True)
            models=[]
            for train_idx,val_idx in splits:
                training=[development[i] for i in train_idx if development[i]['binary_label']==0 or development[i]['events']]
                assert not {r['group'] for r in training}&{development[i]['group'] for i in val_idx}
                models.append((fit(training,mode,c,weak),val_idx))
            for threshold in thresholds:
                predicted=[None]*len(development)
                for model,val_idx in models:
                    for i in val_idx:
                        r=development[i]
                        predicted[i]=model.events(r['x'],r['windows'],threshold)
                report=measure(development,predicted,threshold)
                candidates.append(dict(mode=mode,c=c,weak_negative_weight=weak,threshold=threshold,report=report))
    # Frozen objective: video F1, then timed event recall, fewer false alerts.
    if args.objective=='precision':
        # Development-only selection. Never rank candidates with test outcomes.
        def rank(r):
            m=r['report']; precision=m['precision'] or 0; recall=m['recall'] or 0
            fhalf=1.25*precision*recall/max(1e-12,.25*precision+recall)
            return (m['fp'] <= .25*(m['fp']+m['tn']) and recall >= .5,
                    fhalf,m['matched_within_5_seconds'],-m['predicted_events'],r['threshold'])
        chosen=max(candidates,key=rank)
    else:
        chosen=max(candidates,key=lambda r:(r['report']['f1'],r['report']['matched_within_5_seconds'],
            -r['report']['fp'],r['threshold']))
    write_json(out/'selection.json',dict(candidates=candidates,selected=chosen,
        note='Grouped out-of-fold model selection, not an unbiased final score. Unknown positive regions excluded from fitting.'))
    fitting=[r for r in development if r['binary_label']==0 or r['events']]
    model=fit(fitting,chosen['mode'],chosen['c'],chosen['weak_negative_weight'])
    joblib.dump(model,target)
    metadata=dict(schema_version=1,architecture='resnet18',pooling='mean_max',feature_dim=1024,
        embedding_dim=512,weights_name=extractor.weights_name,weights_sha256=extractor.weights_sha256,
        image_mode='full_frame',sampling=sampling.to_dict())
    metadata.update(pipeline_sha256=sha256(target),threshold=chosen['threshold'],
        classifier='timestamp_supervised_binary_and_multilabel',feature_mode=chosen['mode'],c=chosen['c'],
        weak_negative_weight=chosen['weak_negative_weight'],
        train_ids=[r['row']['id'] for r in fitting],train_sha256=[r['row']['sha256'] for r in fitting],
        workbook_sha256=sha256(workbook),annotations_sha256=sha256(Path(args.annotations)),
        seed=42,review_only=True,test_used_for_training=False,
        created_at=datetime.now(timezone.utc).isoformat(),event_types=list(model.types),
        type_training_video_counts=dict(Counter(t for r in fitting for t in {e['type'] for e in r['events']})),
        research_sha256=sha256(root/'research/rules.json'),
        threshold_objective=args.objective,
        limitations='Small single-camera dataset; visual class hypotheses, not verified legal findings; no fresh external test.')
    write_json(target.with_suffix('.json'),metadata)
    # Only after model/threshold have been frozen do we extract and evaluate test.
    assert not set(metadata['train_sha256']) & {r['row']['sha256'] for r in test}
    test=extract(test,root,cache,extractor,sampling)
    report=measure(test,[model.events(r['x'],r['windows'],chosen['threshold']) for r in test],chosen['threshold'])
    write_json(out/'test_evaluation.json',dict(model_sha256=sha256(target),report=report,
        note='Original held-out video IDs; corrected labels; previously examined internal data, not fresh external test.'))
    gate = release_gate(report)
    write_json(out/'release_gate.json',gate)
    metadata['quality_warning'] = (('Экспериментальная модель: проверка качества не пройдена. '
        if not gate['allowed'] else 'Экспериментальная модель: нужна независимая внешняя проверка. ')+
        f"На {report['tp']+report['fn']} положительных роликах найдены сигналы в {report['tp']}; "
        f"ложные сигналы на {report['fp']} из {report['fp']+report['tn']} нормальных. "
        'Типы действий не подтверждены независимой проверкой.')
    metadata['automatic_release_allowed'] = gate['allowed']
    write_json(target.with_suffix('.json'),metadata)
    print(json.dumps(dict(model=str(target),selected={k:v for k,v in chosen.items() if k!='report'},
        oof={k:v for k,v in chosen['report'].items() if k!='videos'},
        test={k:v for k,v in report.items() if k!='videos'}),ensure_ascii=True),flush=True)


def release_gate(report):
    # A smoke gate, not statistical certification. Small data still requires review.
    checks=dict(precision=(report['precision'] or 0)>=.75,
                recall=(report['recall'] or 0)>=.75,
                distinguishes_normal=report['tn']>0,
                timed_types=report['correct_type_and_time']>=max(1,report['annotated_events']*.5))
    return dict(allowed=all(checks.values()),checks=checks,
        note='Minimum engineering gate only; passing does not establish unseen-camera accuracy or legal validity.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--workbook',required=True)
    parser.add_argument('--annotations',default='data/raw/temporal_annotations_v2.json')
    parser.add_argument('--name',default='anchored_v2')
    parser.add_argument('--objective',choices=['f1','precision'],default='f1')
    run(parser.parse_args())
