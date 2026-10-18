"""Held-video diagnostic for the recovery candidate; never overwrites its weights."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from ml.dataset import sha256, write_json
from ml.detection import ResNetExtractor
from ml.preprocessing import SamplingConfig
from ml.recovery_training import read_descriptions, reconcile, extract, evaluate
from ml.sequence import fit_sequence
from ml.typed import fit


def recording_group(record):
    pairs = {'x-12':'related12_24', 'x-24':'related12_24',
             'x-16':'related16_17', 'x-17':'related16_17',
             'x-22':'related22_23', 'x-23':'related22_23'}
    return pairs.get(record['row']['id'], record['row']['sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workbook', type=Path, required=True)
    parser.add_argument('--mapping', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=Path('models/recovery_typed_v1.joblib'))
    parser.add_argument('--directory', type=Path, default=Path('artifacts/recovery_20261018'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    meta = json.loads(args.model.with_suffix('.json').read_text(encoding='utf-8'))
    before = sha256(args.model)
    if before != meta['pipeline_sha256'] or sha256(args.workbook) != meta['workbook_sha256'] or sha256(args.mapping) != meta['mapping_sha256']:
        raise ValueError('Model or annotation provenance changed')
    manifest = json.loads((root/'artifacts/manifest.json').read_text(encoding='utf-8'))
    if sha256(root/'artifacts/manifest.json') != meta['manifest_sha256']:
        raise ValueError('Manifest changed')
    rows, _ = reconcile(manifest, read_descriptions(args.workbook), json.loads(args.mapping.read_text(encoding='utf-8')))
    training = [r for r in rows if r['row']['id'] in meta['train_ids']]
    assert all(r['row']['split'] in meta.get('fitting_splits', ['train']) and r['row']['split'] != 'test' for r in training)
    extractor = ResNetExtractor(root/'models/resnet18-f37072fd.pth', 'cpu', meta['image_mode'])
    records = extract(training, root, args.directory, extractor, SamplingConfig(**meta['sampling']))
    groups = [recording_group(r) for r in records]
    reports = []
    fitter = fit_sequence if meta['classifier'] == 'sequence' else fit
    for fold, (train_idx, held_idx) in enumerate(GroupKFold(3).split(records, groups=groups), 1):
        tr = [records[i] for i in train_idx]
        held = [records[i] for i in held_idx]
        assert not {r['row']['sha256'] for r in tr} & {r['row']['sha256'] for r in held}
        assert not {recording_group(r) for r in tr} & {recording_group(r) for r in held}
        if not any(r['label']=='normal' for r in tr):
            raise ValueError('Fold has no normal class')
        print('fit held-video fold', fold, flush=True)
        report = evaluate(fitter(tr), held, meta['threshold'])
        reports.append(dict(fold=fold, train_ids=[r['row']['id'] for r in tr],
                            held_ids=[r['row']['id'] for r in held], report=report))
    totals = {k: sum(f['report'][k] for f in reports) for k in ('tp','fp','tn','fn','correctly_typed_positive_videos','typed_positive_videos')}
    tp, fp, fn, tn = (totals[k] for k in ('tp','fp','fn','tn'))
    result = dict(**totals, confusion_matrix=[[tn,fp],[fn,tp]],
        precision=tp/(tp+fp) if tp+fp else None, recall=tp/(tp+fn) if tp+fn else None,
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0., folds=reports,
        note='Grouped training diagnostic, fixed validation-selected architecture/threshold; no tuning on these predictions. Singleton classes may be absent from a fold. Not independent new-camera testing.')
    assert sha256(args.model) == before
    result['model_sha256'] = before
    write_json(args.directory/f'{args.model.stem}_grouped_diagnostic.json', result)
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
