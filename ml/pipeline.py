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
        if metadata.get('image_mode', 'center') not in ('center', 'full_frame', 'multiscale'):
            raise ValueError('Unknown image preprocessing mode')
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
    if metadata.get('image_mode') == 'multiscale':
        from .multiscale import MultiScaleExtractor
        extractor = MultiScaleExtractor(weights, device)
    else:
        extractor = ResNetExtractor(weights, device, metadata.get('image_mode', 'center'))
    if extractor.weights_sha256 != metadata['weights_sha256']:
        raise ForteError('Веса extractor не совпадают с обучением классификатора')
    x, windows, info = extractor.extract(video, SamplingConfig(**metadata['sampling']), deadline)
    probabilities = classifier.predict_proba(x)[:, 1]
    if not np.isfinite(probabilities).all():
        raise ForteError('Модель вернула некорректные оценки')
    events = (classifier.events(x, windows, metadata['threshold'])
              if hasattr(classifier, 'events') else merge_events(windows, probabilities, metadata['threshold']))
    if len(events) > 100:
        raise ForteError('Более 100 событий: сократите видео для ограничения объема evidence')
    # No category has correct-type held-group validation sufficient for automatic
    # findings. Keep every candidate; abstention must not masquerade as accuracy.
    from .triage import route_events
    triage_enabled = hasattr(classifier, 'presence')
    if triage_enabled:
        alerts, review = route_events(events, validated_types=set())
        events = [dict(e, review_only=False) for e in alerts] + [dict(e, review_only=True) for e in review]
    violations = []
    review_candidates = []
    for event in events:
        item = dict(type=event.get('type', GENERIC_TYPE), timecode_sec=event['peak'],
                    description=event.get('description', 'Модель обнаружила подозрительный визуальный фрагмент; тип требует ручной проверки.'),
                    evidence_frames=evidence_frames(video, event, deadline=deadline))
        if event.get('event_type'):
            item['event_type'] = event['event_type']
        context_before = 20. if event.get('event_type') == 'prior_deposit_outside_storage' else 5.
        start = max(0., event['start'] - context_before)
        end = min(info.get('duration_seconds', event['end'] + 5.), event['end'] + 5.)
        item['context_start_sec'] = start
        item['context_end_sec'] = end
        item['description'] += (f' Контекст для просмотра: {start:.1f}–{end:.1f} с; '
                                f'выделенный момент — {event["peak"]:.1f} с. '
                                'Сравните действия до этого момента и после него. '
                                'Границы фрагмента определены моделью, а не подтверждены разметкой.')
        if event.get('review_only'):
            item['review_reason'] = event.get('review_reason') or 'Гипотеза не прошла проверку типа на отложенных видео.'
            item['review_priority'] = event.get('review_priority', 'normal')
            review_candidates.append(item)
        else:
            violations.append(item)
    check_deadline(deadline)
    elapsed = perf_counter()-started
    result = dict(verdict='violation' if violations else 'no_violation', violations=violations,
                elapsed_seconds=elapsed, within_five_minutes=elapsed < 300)
    if triage_enabled:
        result.update(review_candidates=review_candidates, review_required=bool(review_candidates),
            assessment_status='needs_review' if review_candidates else 'incomplete',
            candidate_count=len(violations)+len(review_candidates))
    if hasattr(classifier, 'presence'):
        result.update(analysis_backend='classifier_with_action_supervision', analysis_complete=False,
            workflow_checks=[dict(rule_id='visual_coverage', state='inconclusive',title='Полнота проверки',
                note='Наличие кассира проверяется отдельной моделью. Удостоверение, ключ и все действия пока не распознаются надёжно.')],
            analysis_limitations=['Отсутствие найденных событий не доказывает отсутствие нарушений.',
                'Модель обучена на небольшой коллекции одной комнаты; качество на новом ракурсе не подтверждено.',
                'Предъявление удостоверения остаётся гипотезой категорийной модели, а не результатом детектора документа.'])
    if hasattr(classifier, 'customer_presence'):
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
