"""Video-isolated train/validation/test and validation-only threshold selection."""
import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import ForteError, prepare, sha256, validate_manifest, write_json
from .detection import ResNetExtractor
from .preprocessing import SamplingConfig
from .temporal import merge_events, window_labels


def build_records(rows, videos, extractor, sampling, labeling, budget=285):
    records = []
    for row in rows:
        started = perf_counter()
        print(f"extract {row['split']}: {row['id']}", flush=True)
        x, windows, info = extractor.extract(videos/row['file'], sampling, started+budget)
        events = row.get('timestamps', [] if row['timestamp'] is None else [row['timestamp']])
        if any(t >= info['duration_seconds'] for t in events):
            raise ForteError(f"Timestamp вне видео: {row['file']}")
        y, weak = window_labels(windows, row['label'], events, **labeling)
        if row['label'] and not (y == 1).any():
            raise ForteError(f"Нет положительных окон для timestamp: {row['file']}")
        records.append(dict(row=row, x=x, y=y, weak=weak, windows=windows, elapsed_seconds=perf_counter()-started))
    return records


def binary_metrics(y, probabilities, threshold):
    if not len(y):
        return {'available': False, 'reason': 'Нет размеченных примеров'}
    pred = np.asarray(probabilities) >= threshold
    positive_count = int(np.sum(np.asarray(y) == 1))
    predicted_count = int(pred.sum())
    return dict(available=True, count=len(y), precision=float(precision_score(y, pred, zero_division=0)) if predicted_count else None,
                recall=float(recall_score(y, pred, zero_division=0)) if positive_count else None,
                f1=float(f1_score(y, pred, zero_division=0)) if positive_count+predicted_count else None,
                confusion_matrix=confusion_matrix(y, pred, labels=[0, 1]).tolist(), labels=[0, 1],
                roc_auc=float(roc_auc_score(y, probabilities)) if len(set(y)) == 2 else None)


def predict_records(pipeline, records):
    return [pipeline.predict_proba(r['x'])[:, 1] for r in records]


def tune_threshold(records, probabilities):
    y = [r['row']['label'] for r in records]
    if set(y) != {0, 1}:
        raise ForteError('Threshold tuning требует оба класса в validation')
    video_scores = np.asarray([p.max() for p in probabilities])
    candidates = np.unique(np.r_[0.05, np.arange(0.1, 1.0, 0.05), video_scores])
    candidates = candidates[(candidates > 0) & (candidates < 1)]
    # Predeclared objective: video F1. Ties choose the higher threshold.
    return float(max(candidates, key=lambda t: (f1_score(y, video_scores >= t, zero_division=0), t)))


def report_metrics(records, probabilities, threshold):
    true_windows, predicted_windows, weak_y, weak_p = [], [], [], []
    errors, detected_positive, positives, details = [], 0, 0, []
    annotated_events, matched_events = 0, 0
    for r, p in zip(records, probabilities):
        known = (r['y'] >= 0) & ~r['weak']
        true_windows.extend(r['y'][known].tolist())
        predicted_windows.extend(p[known].tolist())
        weak_y.extend(r['y'][r['weak']].tolist())
        weak_p.extend(p[r['weak']].tolist())
        events = merge_events(r['windows'], p, threshold)
        error = None
        if r['row']['label']:
            positives += 1
            annotations = r['row'].get('timestamps', [r['row']['timestamp']])
            annotated_events += len(annotations)
            if events:
                detected_positive += 1
                event_errors = [min(abs(e['peak']-t) for e in events) for t in annotations]
                errors.extend(event_errors)
                matched_events += len(event_errors)
                error = float(np.mean(event_errors))
        details.append(dict(id=r['row']['id'], label=r['row']['label'], score=float(p.max()),
                            localization_error_seconds=error, elapsed_seconds=r['elapsed_seconds']))
    return dict(threshold=threshold,
                window_metrics=binary_metrics(true_windows, predicted_windows, threshold),
                weak_negative_metrics=binary_metrics(weak_y, weak_p, threshold),
                video_metrics=binary_metrics([r['row']['label'] for r in records], [float(p.max()) for p in probabilities], threshold),
                localization=dict(mean_absolute_error_seconds=float(np.mean(errors)) if errors else None,
                                  evaluated_positive_videos=detected_positive, total_positive_videos=positives,
                                  missed_positive_videos=positives-detected_positive,
                                  total_annotated_events=annotated_events, evaluated_events=matched_events,
                                  definition='Nearest predicted peak per annotation; conditional on a positive video having a prediction. Not one-to-one event matching.'),
                videos=details)


def train(manifest_path, videos, model_path, weights, sampling, labeling, device='cpu', seed=42):
    if model_path.exists() or model_path.with_suffix('.json').exists():
        raise ForteError('Модель уже существует; используйте новый --model для нового эксперимента')
    random.seed(seed)
    np.random.seed(seed)
    rows = validate_manifest(json.loads(manifest_path.read_text(encoding='utf-8')), videos)
    extractor = ResNetExtractor(weights, device)
    extractor.torch.manual_seed(seed)
    train_rows = [r for r in rows if r['split'] == 'train']
    validation_rows = [r for r in rows if r['split'] == 'validation']
    # Deliberately no extraction of test or excluded rows.
    training = build_records(train_rows, videos, extractor, sampling, labeling)
    validation = build_records(validation_rows, videos, extractor, sampling, labeling)
    x = np.concatenate([r['x'][r['y'] >= 0] for r in training])
    y = np.concatenate([r['y'][r['y'] >= 0] for r in training])
    weak = np.concatenate([r['weak'][r['y'] >= 0] for r in training])
    if set(y) != {0, 1}:
        raise ForteError('Для обучения нужны оба класса окон')
    pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', LogisticRegression(class_weight='balanced', C=0.1, max_iter=3000, random_state=seed))])
    pipeline.fit(x, y, classifier__sample_weight=np.where(weak, 0.25, 1.0))
    p = predict_records(pipeline, validation)
    threshold = tune_threshold(validation, p)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = model_path.with_suffix('.tmp')
    joblib.dump(pipeline, temporary)
    temporary.replace(model_path)
    import sklearn
    import torch
    import torchvision
    metadata = dict(schema_version=1, architecture='resnet18', pooling='mean_max', embedding_dim=512,
                    feature_dim=1024, weights_name='IMAGENET1K_V1', weights_sha256=extractor.weights_sha256,
                    pipeline_sha256=sha256(model_path), sampling=sampling.to_dict(), labeling=labeling,
                    threshold=threshold, threshold_objective='validation_video_f1', seed=seed,
                    created_at=datetime.now(timezone.utc).isoformat(), manifest_sha256=sha256(manifest_path),
                    train_ids=[r['id'] for r in train_rows], train_sha256=[r['sha256'] for r in train_rows],
                    validation_ids=[r['id'] for r in validation_rows], training_windows=len(y), weak_negative_windows=int(weak.sum()),
                    versions=dict(sklearn=sklearn.__version__, torch=torch.__version__, torchvision=torchvision.__version__),
                    validation_metrics=report_metrics(validation, p, threshold), test_used_for_training=False)
    write_json(model_path.with_suffix('.json'), metadata)
    return metadata


def evaluate(manifest_path, videos, model_path, weights, device='cpu'):
    from .pipeline import load_classifier
    pipeline, metadata = load_classifier(model_path)
    if metadata['manifest_sha256'] != sha256(manifest_path):
        raise ForteError('Manifest изменился после обучения')
    rows = validate_manifest(json.loads(manifest_path.read_text(encoding='utf-8')), videos)
    test = [r for r in rows if r['split'] == 'test']
    if set(r['sha256'] for r in test) & set(metadata['train_sha256']):
        raise ForteError('Обнаружена утечка test в train')
    extractor = ResNetExtractor(weights, device)
    if extractor.weights_sha256 != metadata['weights_sha256']:
        raise ForteError('Веса extractor отличаются от обучения')
    records = build_records(test, videos, extractor, SamplingConfig(**metadata['sampling']), metadata['labeling'])
    return dict(split='test', metrics=report_metrics(records, predict_records(pipeline, records), metadata['threshold']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'train', 'evaluate'])
    parser.add_argument('--labels', type=Path, default=Path('data/raw/open_labels.xlsx'))
    parser.add_argument('--videos', type=Path, default=Path('data/videos'))
    parser.add_argument('--manifest', type=Path, default=Path('artifacts/manifest.json'))
    parser.add_argument('--model', type=Path, default=Path('models/classifier.joblib'))
    parser.add_argument('--weights', type=Path, default=Path('models/resnet18-f37072fd.pth'))
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--fps', type=float, default=2)
    parser.add_argument('--window', type=float, default=4)
    parser.add_argument('--stride', type=float, default=2)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--positive-radius', type=float, default=1)
    parser.add_argument('--exclusion-zone', type=float, default=6)
    parser.add_argument('--weak-negatives', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('artifacts/test_metrics.json'))
    args = parser.parse_args()
    try:
        if args.command == 'prepare':
            result = prepare(args.labels, args.videos, args.manifest)
        elif args.command == 'train':
            if not 0 <= args.positive_radius < args.exclusion_zone:
                raise ForteError('Нужно 0 <= positive-radius < exclusion-zone')
            sampling = SamplingConfig(args.fps, args.window, args.stride, args.batch_size)
            labeling = dict(positive_radius=args.positive_radius, exclusion_zone=args.exclusion_zone, weak_negatives=args.weak_negatives)
            result = train(args.manifest, args.videos, args.model, args.weights, sampling, labeling, args.device)
        else:
            result = evaluate(args.manifest, args.videos, args.model, args.weights, args.device)
            write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=True, indent=2, allow_nan=False))
    except (ForteError, OSError, ValueError, KeyError) as exc:
        parser.exit(2, f'{ascii(str(exc))}\n')


if __name__ == '__main__':
    main()
