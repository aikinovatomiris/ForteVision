import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np

from .dataset import ForteError, sha256
from .detection import ResNetExtractor
from .evidence import evidence_frames
from .preprocessing import SamplingConfig, check_deadline
from .rules import GENERIC_TYPE
from .temporal import merge_events


def load_classifier(path: Path):
    path = Path(path)
    if not path.is_file() or not path.with_suffix('.json').is_file():
        raise ForteError('Классификатор не обучен: нужны classifier.joblib и metadata JSON. Выполните python -m ml.training train.')
    try:
        metadata = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        if metadata['schema_version'] != 1 or metadata['architecture'] != 'resnet18' or metadata['pooling'] != 'mean_max' or metadata['feature_dim'] != 1024:
            raise ValueError('Несовместимая конфигурация модели')
        if not 0 < metadata['threshold'] < 1 or not metadata['train_sha256']:
            raise ValueError('Неверный threshold или отсутствует provenance')
        SamplingConfig(**metadata['sampling'])
        if sha256(path) != metadata['pipeline_sha256']:
            raise ValueError('SHA-256 classifier.joblib не совпадает')
        # joblib can execute code: load only administrator-provided local artifacts.
        pipeline = joblib.load(path)
        if pipeline.n_features_in_ != 1024 or list(pipeline.classes_) != [0, 1]:
            raise ValueError('Неверные классы или размер признаков')
        return pipeline, metadata
    except Exception as exc:
        raise ForteError(f'Некорректная локальная модель: {exc}') from exc


def analyze(video: Path, model: Path, weights: Path, device='cpu', budget_seconds=285):
    started = perf_counter()
    deadline = started + budget_seconds
    classifier, metadata = load_classifier(model)
    extractor = ResNetExtractor(weights, device)
    if extractor.weights_sha256 != metadata['weights_sha256']:
        raise ForteError('Веса extractor не совпадают с обучением классификатора')
    x, windows, info = extractor.extract(video, SamplingConfig(**metadata['sampling']), deadline)
    probabilities = classifier.predict_proba(x)[:, 1]
    if not np.isfinite(probabilities).all():
        raise ForteError('Модель вернула некорректные оценки')
    events = merge_events(windows, probabilities, metadata['threshold'])
    if len(events) > 100:
        raise ForteError('Более 100 событий: сократите видео для ограничения объема evidence')
    violations = []
    for event in events:
        violations.append(dict(type=GENERIC_TYPE, timecode_sec=event['peak'],
                               description='Модель обнаружила подозрительный визуальный фрагмент; тип требует ручной проверки.',
                               evidence_frames=evidence_frames(video, event, deadline=deadline)))
    check_deadline(deadline)
    elapsed = perf_counter()-started
    return dict(verdict='violation' if violations else 'no_violation', violations=violations,
                elapsed_seconds=elapsed, within_five_minutes=elapsed < 300)
