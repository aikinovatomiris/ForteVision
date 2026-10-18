"""Lossless candidate routing; scores are uncalibrated, not probabilities of guilt."""
from copy import deepcopy
import math

ALERT_THRESHOLD = .80  # fixed before evaluating grouped predictions
REVIEW_THRESHOLD = .35

def route_events(events, alert_threshold=ALERT_THRESHOLD, validated_types=None):
    alerts, review = [], []
    for source in events:
        event = deepcopy(source)
        score = event.get('score', 0.)
        if not isinstance(score, (int, float)) or not math.isfinite(score):
            score = 0.
        high = bool(event.get('event_type')) and score >= alert_threshold and not event.get('review_only')
        if validated_types is not None:
            high = high and event.get('event_type') in validated_types
        event['review_priority'] = 'high' if score >= alert_threshold else 'normal'
        event['review_reason'] = (None if high else
            'Тип события или оценка модели недостаточны для автоматической тревоги. Фрагмент сохранён для просмотра.')
        (alerts if high else review).append(event)
    assert len(alerts) + len(review) == len(events)
    return alerts, review

def add_review_candidates(primary, sensitive):
    """Keep every primary interval verbatim; append additional non-covered intervals."""
    result=deepcopy(primary)
    for candidate in sensitive:
        covered=any(e.get('event_type')==candidate.get('event_type') and
                    e['start']<=candidate['start'] and e['end']>=candidate['end'] for e in result)
        if not covered:
            extra=deepcopy(candidate)
            extra['review_only']=True
            result.append(extra)
    return sorted(result,key=lambda e:e['peak'])
