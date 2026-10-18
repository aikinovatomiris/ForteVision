"""One final development refit after validation selection, before final test evaluation."""
import argparse
import json
from pathlib import Path

import joblib

from ml.dataset import sha256, write_json
from ml.detection import ResNetExtractor
from ml.preprocessing import SamplingConfig
from ml.recovery_training import read_descriptions, reconcile, extract, evaluate
from ml.sequence import fit_sequence
from ml.typed import fit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workbook',type=Path,required=True)
    parser.add_argument('--mapping',type=Path,required=True)
    parser.add_argument('--source',type=Path,default=Path('models/recovery_typed_v1.joblib'))
    parser.add_argument('--model',type=Path,default=Path('models/recovery_review_v1.joblib'))
    parser.add_argument('--directory',type=Path,default=Path('artifacts/recovery_20261018'))
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    if args.model.exists() or args.model.with_suffix('.json').exists():raise ValueError('Refusing overwrite')
    meta=json.loads(args.source.with_suffix('.json').read_text(encoding='utf-8'))
    if sha256(args.source)!=meta['pipeline_sha256'] or sha256(args.workbook)!=meta['workbook_sha256'] or sha256(args.mapping)!=meta['mapping_sha256']:
        raise ValueError('Source provenance changed')
    manifest=json.loads((root/'artifacts/manifest.json').read_text(encoding='utf-8'))
    if sha256(root/'artifacts/manifest.json')!=meta['manifest_sha256']:raise ValueError('Manifest changed')
    records,_=reconcile(manifest,read_descriptions(args.workbook),json.loads(args.mapping.read_text(encoding='utf-8')))
    development=[r for r in records if r['row']['split'] in ('train','validation') and r['label'] and not r['excluded_from_fitting']]
    test_hashes={r['row']['sha256'] for r in records if r['row']['split']=='test'}
    assert not test_hashes & {r['row']['sha256'] for r in development}
    plan=dict(source_model_sha256=meta['pipeline_sha256'],architecture=meta['classifier'],
        threshold=meta['threshold'],development_ids=[r['row']['id'] for r in development],
        rationale='Final refit includes development types absent from original train. No test-based selection.',
        test_used_for_training=False,validation_no_longer_independent=True)
    write_json(args.directory/'release_refit_plan.json',plan)
    extractor=ResNetExtractor(root/'models/resnet18-f37072fd.pth','cpu',meta['image_mode'])
    training=extract(development,root,args.directory,extractor,SamplingConfig(**meta['sampling']))
    model=(fit_sequence if meta['classifier']=='sequence' else fit)(training)
    joblib.dump(model,args.model)
    from collections import Counter
    meta.update(pipeline_sha256=sha256(args.model),selection_model_sha256=plan['source_model_sha256'],
        selection_validation_ids=meta['validation_ids'],validation_ids=[],
        train_ids=[r['row']['id'] for r in training],train_sha256=[r['row']['sha256'] for r in training],
        fitting_splits=['train','validation'],event_types=[str(c) for c in model.estimator.classes_ if c!='normal'],
        training_video_counts=dict(Counter(r['label'] for r in training)),
        threshold_objective='Transferred unchanged from validation-selected candidate; not recalibrated after refit',
        validation_metrics_apply_to='selection_model_only_not_this_refit')
    meta.pop('validation_metrics',None)
    write_json(args.model.with_suffix('.json'),meta)
    regression=evaluate(model,[r for r in training if r['row']['id']=='x-4'],meta['threshold'])
    write_json(args.directory/'release_x4_regression.json',dict(report=regression,note='Training example, not generalization evidence'))
    print(json.dumps(dict(model=str(args.model),training_counts=meta['training_video_counts'],x4=regression),ensure_ascii=True,indent=2))


if __name__=='__main__':main()
