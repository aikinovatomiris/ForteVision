"""Human corrections are pending annotations, never automatic training labels."""
import hashlib
import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path

DECISIONS = {
    'false_positive': 'Ложная тревога в указанном фрагменте',
    'missed_event': 'Модель пропустила действие',
    'wrong_type': 'Неверно названо действие',
    'wrong_time': 'Неверный таймкод',
    'confirmed': 'Действие и интервал подтверждаю',
    'normal_video': 'Просмотрена вся запись: нарушений не замечено',
    'uncertain': 'Не могу определить по записи',
}


def make_feedback(video_bytes, filename, job_id, decision, start, end, observed,
                  event_type, provenance, watched_full=False, model_identity=None,
                  research_rule=None, conditions_review='not_checked'):
    if not video_bytes:
        raise ValueError('Нужно исходное видео этой сессии')
    if decision not in DECISIONS or provenance not in ('development', 'jury_test', 'unknown'):
        raise ValueError('Неизвестный вид исправления или происхождение видео')
    if not all(math.isfinite(t) and t >= 0 for t in (start, end)) or end <= start:
        raise ValueError('Конец интервала должен быть больше начала; укажите секунды от начала видео')
    if not 5 <= len(observed.strip()) <= 3000 or len(event_type.strip()) > 200:
        raise ValueError('Опишите наблюдаемое действие (5–3000 символов), тип — до 200 символов')
    if decision == 'normal_video' and not watched_full:
        raise ValueError('Для отрицательной метки всей записи нужно просмотреть её полностью')
    if conditions_review not in ('not_checked', 'observed', 'absent', 'not_visible'):
        raise ValueError('Неизвестный статус проверки условий')
    rule_context = None
    if research_rule:
        rule_context = {k: research_rule.get(k) for k in (
            'rule_id', 'event_type', 'event_name_ru', 'source_url', 'source_clause',
            'rule_version', 'detection_conditions', 'evidence_requirements', 'confidence_limitations')}
        rule_context.update(role='human_selected_research_candidate', conditions_review=conditions_review)
    return dict(schema_version=1, id=uuid.uuid4().hex,
        created_at=datetime.now(timezone.utc).isoformat(), status='pending_review',
        video_sha256=hashlib.sha256(video_bytes).hexdigest(),
        filename=filename.replace('\\', '/').rsplit('/', 1)[-1], job_id=job_id,
        decision=decision, start_sec=float(start), end_sec=float(end),
        observed_action=observed.strip(), proposed_type=event_type.strip() or None,
        watched_full=bool(watched_full), scope='video' if decision == 'normal_video' else 'interval',
        provenance=provenance, eligible_for_training=False,
        training_block_reason='Requires human adjudication and frozen split check; jury/unknown data must not enter training.',
        model_identity=model_identity, source='user_feedback', research_rule=rule_context)


def save_feedback(record, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{uuid.UUID(record["id"]).hex}.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return path
