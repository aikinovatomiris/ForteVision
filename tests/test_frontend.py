"""UI navigation and states, without model training or external requests."""
import base64
import json
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def ui():
    response = Mock(status_code=200)
    response.json.return_value = {'status': 'ok'}
    with patch('requests.Session.get', return_value=response), patch('requests.Session.post') as post:
        app = AppTest.from_file(str(ROOT/'app/frontend.py'), default_timeout=15).run()
        assert not app.exception
        yield app, response
        post.assert_not_called()


def test_research_navigation_search_and_filter(ui):
    app, _ = ui
    # Catalog is optional on installations without local research artifacts.
    if not (ROOT/'research/rules.json').exists():
        pytest.skip('Optional local research catalog is absent')
    catalog = json.loads((ROOT/'research/rules.json').read_text(encoding='utf-8-sig'))
    app.radio[0].set_value('Нормативные акты').run()
    assert not app.exception
    assert len(app.subheader) == len(catalog['rules'])
    selected = catalog['rules'][0]
    app.text_input[0].set_value(selected['rule_id']).run()
    assert len(app.subheader) == 1
    assert app.subheader[0].value == selected['event_name_ru']
    app.text_input[0].set_value('zzzz-no-match').run()
    assert not app.subheader
    app.text_input[0].set_value('').run()
    app.selectbox[0].set_value(selected['source_clause']).run()
    assert len(app.subheader) == sum(r['source_clause'] == selected['source_clause'] for r in catalog['rules'])
    assert not app.exception


def test_research_missing_file_is_explained(ui):
    app, _ = ui
    original = Path.read_text
    def read(path, *args, **kwargs):
        if path == ROOT/'research/rules.json':
            raise FileNotFoundError('Catalog missing')
        return original(path, *args, **kwargs)
    with patch.object(Path, 'read_text', read):
        app.radio[0].set_value('Нормативные акты').run()
    assert not app.exception
    assert any('research/rules.json' in w.value for w in app.warning)


def test_processing_and_failed_states(ui):
    app, response = ui
    app.session_state['job_id'] = 'test-job'
    response.json.return_value = {'job_id': 'test-job', 'status': 'processing'}
    app.run()
    assert not app.exception
    assert len(app.status) == 1
    response.json.return_value = {'job_id': 'test-job', 'status': 'failed', 'error': 'Модель недоступна'}
    app.run()
    assert not app.exception
    assert len(app.error) == 1


@pytest.mark.parametrize('positive', [True, False])
def test_completed_result_rendering(ui, positive):
    app, _ = ui
    ok, encoded = cv2.imencode('.jpg', np.zeros((32, 32, 3), np.uint8))
    assert ok
    event = dict(type='Признаки возможного нарушения', timecode_sec=12.5,
                 description='Требуется ручная проверка.',
                 evidence_frames=[dict(timestamp_sec=12.5, image_base64=base64.b64encode(encoded).decode())])
    app.session_state['job_id'] = 'test-job'
    app.session_state['result'] = dict(job_id='test-job', status='done',
        verdict='violation' if positive else 'no_violation', violations=[event] if positive else [], elapsed_seconds=1.5)
    app.run()
    assert not app.exception
    assert len(app.get('imgs')) == int(positive)
    assert len(app.metric) == 3
    assert len(app.get('download_button')) == 1

def test_event_playback_uses_matching_uploaded_video(ui):
    app,_=ui
    ok,encoded=cv2.imencode('.jpg',np.zeros((32,32,3),np.uint8))
    event=dict(type='Тип события не определён',description='Не выбран тип.',timecode_sec=12.,context_start_sec=5.,context_end_sec=20.,evidence_frames=[dict(timestamp_sec=12.,image_base64=base64.b64encode(encoded).decode())])
    app.session_state['job_id']='context-job'
    app.session_state['result']=dict(job_id='context-job',status='done',verdict='violation',violations=[event])
    app.session_state['analyzed_video_bytes']=b'video-fixture'
    app.session_state['analyzed_video_mime']='video/mp4'
    app.session_state['analyzed_video_job_id']='context-job'
    with patch('streamlit.video') as player:
        app.run()
        assert not app.exception
        player.assert_called_once_with(b'video-fixture',format='video/mp4',start_time=5.,end_time=20.)
    app.session_state['analyzed_video_job_id']='different-job'
    with patch('streamlit.video') as player:
        app.run()
        assert not app.exception
        player.assert_not_called()
    assert any('Исходное видео недоступно' in item.value for item in app.info)


def test_workflow_unknown_does_not_render_clean_bill(ui):
    app,_=ui
    app.session_state['job_id']='workflow-test'
    app.session_state['result']=dict(job_id='workflow-test',status='done',verdict='no_violation',violations=[],analysis_complete=False,
        workflow_checks=[dict(title='Передача документа',state='unknown',timestamps=[],note='Не видно')],
        observations=[dict(timestamp_sec=1.5,detail='Деньги на столе')])
    app.run()
    assert not app.exception
    assert any('Есть непроверенные шаги операции' in x.value for x in app.markdown)
    assert any('Не удалось проверить' in x.value for x in app.markdown)


def test_review_result_keeps_candidate_and_never_shows_clean_banner(ui):
    app, _ = ui
    ok, encoded = cv2.imencode('.jpg', np.zeros((32, 32, 3), np.uint8))
    assert ok
    event = dict(type='Документ требует проверки', timecode_sec=12., description='Гипотеза.',
                 review_reason='Тип не подтверждён на отложенных видео.', review_priority='high',
                 evidence_frames=[dict(timestamp_sec=12., image_base64=base64.b64encode(encoded).decode())])
    app.session_state['job_id'] = 'review-job'
    app.session_state['result'] = dict(job_id='review-job', status='done', verdict='no_violation', violations=[],
        review_candidates=[event], review_required=True, assessment_status='needs_review', analysis_complete=False)
    app.run()
    assert not app.exception
    assert any('Гипотеза: Документ требует проверки' in h.value for h in app.subheader)
    rendered=' '.join(item.value for item in app.markdown)
    assert 'Нужна проверка выделенных моментов' in rendered
    assert 'Нарушение не обнаружено' not in rendered


def test_quality_warning_visible_without_workflow_checks(ui):
    app,_=ui
    app.session_state['job_id']='quality-job'
    app.session_state['result']=dict(job_id='quality-job',status='done',verdict='no_violation',
        violations=[],analysis_complete=False,model_info={'quality_warning':'Проверка качества не пройдена'},
        analysis_limitations=['Мало независимых примеров'])
    app.run()
    assert not app.exception
    assert any('Проверка качества не пройдена' in item.value for item in app.warning)
    assert any('Мало независимых примеров' in item.value for item in app.caption)


def test_binary_capabilities_and_local_feedback(ui, tmp_path):
    from types import SimpleNamespace
    app, _ = ui
    app.session_state['job_id'] = 'feedback-job'
    app.session_state['result'] = dict(job_id='feedback-job', status='done', verdict='no_violation',
        violations=[], analysis_complete=False, video_duration_seconds=10.,
        model_info=dict(artifact='classifier.joblib', sha256='test', supports_event_types=False))
    app.session_state['analyzed_video_bytes'] = b'fixture-video'
    app.session_state['analyzed_video_job_id'] = 'feedback-job'
    app.run()
    assert any('Распознавание конкретных действий' in w.value for w in app.warning)
    app.text_area[0].set_value('Здесь кассир не покидает рабочее место')
    button = next(b for b in app.button if b.label == 'Сохранить исправление локально')
    with patch('app.config.Settings', return_value=SimpleNamespace(job_db=tmp_path/'jobs.sqlite3')):
        button.click().run()
    assert not app.exception
    records = list((tmp_path/'feedback').glob('*.json'))
    assert len(records) == 1
    saved = json.loads(records[0].read_text(encoding='utf-8'))
    assert saved['provenance'] == 'unknown' and saved['eligible_for_training'] is False


def test_priority_candidate_shown_first_without_dropping_uncertain_interval(ui):
    app, _ = ui
    _, encoded = cv2.imencode('.jpg', np.zeros((16,16,3), np.uint8))
    frame = dict(timestamp_sec=54., image_base64=base64.b64encode(encoded).decode())
    unknown = dict(type='Тип события не определён',timecode_sec=54.,description='Неясный фрагмент',
                   review_reason='Требует просмотра',review_priority='normal',evidence_frames=[frame])
    typed = dict(unknown,type='Деньги предыдущего клиента не убраны',timecode_sec=58.,review_priority='high',
                 event_type='prior_deposit_outside_storage')
    app.session_state['job_id']='priority-job'
    app.session_state['result']=dict(job_id='priority-job',status='done',verdict='no_violation',
        violations=[],review_candidates=[unknown,typed],review_required=True,assessment_status='needs_review',analysis_complete=False)
    app.run()
    assert not app.exception
    assert app.subheader[0].value=='Гипотеза: Деньги предыдущего клиента не убраны'
    assert app.subheader[1].value=='Гипотеза: Тип события не определён'
    assert len(app.get('imgs'))==2
