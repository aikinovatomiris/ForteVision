import json
from pathlib import Path

import pytest

from app.review_feedback import make_feedback, save_feedback
from scripts.audit_active_model import threshold_frontier


def correction(**kwargs):
    args = dict(video_bytes=b'local video', filename='../video.mov', job_id='job',
                decision='false_positive', start=1., end=3., observed='Кассир остаётся на месте',
                event_type='', provenance='development')
    args.update(kwargs)
    return make_feedback(**args)


def test_interval_negative_never_becomes_video_label_or_training_data(tmp_path):
    record = correction()
    assert record['scope'] == 'interval' and record['status'] == 'pending_review'
    assert record['eligible_for_training'] is False and record['filename'] == 'video.mov'
    path = save_feedback(record, tmp_path)
    assert json.loads(path.read_text(encoding='utf-8')) == record
    assert 'local video' not in path.read_text(encoding='utf-8')
    with pytest.raises(FileExistsError):
        save_feedback(record, tmp_path)


@pytest.mark.parametrize('provenance', ['jury_test', 'unknown', 'development'])
def test_feedback_always_requires_adjudication(provenance):
    assert correction(provenance=provenance)['eligible_for_training'] is False


@pytest.mark.parametrize('changes', [dict(start=float('nan')), dict(end=.5),
    dict(observed=''), dict(decision='normal_video', watched_full=False)])
def test_invalid_feedback_rejected(changes):
    with pytest.raises(ValueError):
        correction(**changes)


def test_full_normal_requires_full_review():
    assert correction(decision='normal_video', watched_full=True)['scope'] == 'video'


def test_research_context_is_human_selected_not_model_prediction():
    record = correction(research_rule={'rule_id': 'R01', 'event_type': 'cashier_left_active_service',
                                     'source_clause': '22'}, conditions_review='not_visible')
    assert record['research_rule']['role'] == 'human_selected_research_candidate'
    assert record['research_rule']['conditions_review'] == 'not_visible'
    assert record['eligible_for_training'] is False


def test_threshold_cannot_repair_reversed_scores():
    points = threshold_frontier([{'label': 1, 'score': .1}, {'label': 0, 'score': .9}])
    assert all(p['fp'] == 1 for p in points if p['fn'] == 0)
    assert any(p['tp'] == 0 and p['fp'] == 0 for p in points)
