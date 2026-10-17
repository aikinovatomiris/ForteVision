import base64
import json
import sqlite3
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api import create_app
from app.config import Settings
from app.jobs import JobExpiredError, JobStore
from app.rule_links import attach_verified_sources
from app.schemas import EvidenceFrame
from ml.dataset import sha256


@pytest.fixture
def settings(tmp_path):
    return Settings(api_key='tests-only', upload_dir=tmp_path/'uploads', model=tmp_path/'model.joblib',
                    weights=tmp_path/'weights.pth', timeout_seconds=2, max_jobs=2)


def wait_terminal(store, key):
    deadline = time.monotonic()+4
    while time.monotonic() < deadline:
        result = store.get(key)
        if result['status'] != 'processing':
            return result
        time.sleep(.02)
    raise AssertionError('Job remained processing')


def video(settings, name):
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    path = settings.upload_dir/name
    path.write_bytes(b'fixture')
    return path


def test_completed_jobs_do_not_block_new_uploads_and_survive_restart(settings):
    store = JobStore(settings, lambda _: dict(verdict='no_violation', violations=[]))
    store.start()
    try:
        for i in range(6):
            key = store.submit(video(settings, f'{i}.mov'))
            assert wait_terminal(store, key)['status'] == 'done'
        assert len(store.jobs) <= settings.max_jobs
    finally:
        store.stop()
    restored = JobStore(settings)
    restored.start()
    try:
        assert restored.get(key)['status'] == 'done'
        assert restored.delete(key)
        assert restored.get(key) is None
    finally:
        restored.stop()


def test_expired_upload_is_not_accepted(settings):
    store = JobStore(settings)
    with pytest.raises(JobExpiredError):
        store.submit(video(settings, 'expired.mov'), time.monotonic()-3)


def test_deadline_includes_waiting_and_late_result_cannot_replace_failure(settings):
    started, release = threading.Event(), threading.Event()
    calls = []
    def runner(path):
        calls.append(path.name)
        started.set()
        release.wait(3)
        return dict(verdict='no_violation', violations=[])
    store = JobStore(settings, runner)
    store.start()
    try:
        first = store.submit(video(settings, 'active.mov'))
        assert started.wait(1)
        second = store.submit(video(settings, 'queued.mov'), time.monotonic()-1.8)
        assert wait_terminal(store, second)['status'] == 'failed'
        assert wait_terminal(store, first)['status'] == 'failed'
        release.set()
        store.queue.join()
        assert store.get(first)['status'] == 'failed'
        assert calls == ['active.mov']
        assert not list(settings.upload_dir.iterdir())
    finally:
        release.set()
        store.stop()


def test_remaining_budget_is_passed_to_subprocess(settings, monkeypatch):
    budgets = []
    def runner(path, config, remaining):
        budgets.append(remaining)
        return dict(verdict='no_violation', violations=[])
    monkeypatch.setattr('app.jobs.subprocess_runner', runner)
    store = JobStore(settings)
    store.start()
    try:
        key = store.submit(video(settings, 'budget.mov'), time.monotonic()-1)
        result = wait_terminal(store, key)
        assert result['status'] == 'done'
        assert 0 < budgets[0] <= 1
        assert result['elapsed_seconds'] >= 1
    finally:
        store.stop()


def test_recovery_marks_interrupted_jobs_failed(settings):
    settings.job_db.parent.mkdir(parents=True)
    with sqlite3.connect(settings.job_db) as db:
        db.execute('CREATE TABLE jobs (id TEXT PRIMARY KEY, payload TEXT NOT NULL, completed REAL)')
        db.execute('INSERT INTO jobs VALUES (?,?,NULL)', ('interrupted', json.dumps(dict(job_id='interrupted', status='processing'))))
    store = JobStore(settings)
    store.start()
    try:
        assert store.get('interrupted')['status'] == 'failed'
    finally:
        store.stop()


def test_second_process_cannot_own_store(settings):
    store = JobStore(settings)
    store.start()
    try:
        with pytest.raises(RuntimeError, match='один процесс'):
            JobStore(settings).start()
    finally:
        store.stop()


def test_readiness_and_delete_authentication(settings):
    with TestClient(create_app(settings)) as client:
        assert client.get('/health').status_code == 200
        assert client.get('/ready').status_code == 503
        assert client.delete('/api/v1/jobs/00000000-0000-0000-0000-000000000000').status_code == 401


def test_invalid_jpeg_is_rejected():
    with pytest.raises(ValidationError):
        EvidenceFrame(timestamp_sec=1, image_base64=base64.b64encode(b'not a JPEG').decode())


def test_sources_require_matching_model_validation_and_specific_type(settings, tmp_path):
    settings.model.write_bytes(b'fixture-model')
    (tmp_path/'validation.json').write_text('{}')
    rule = dict(event_type='fixture-event', source_title='TEST FIXTURE ONLY', source_url='https://adilet.zan.kz/rus/docs/fixture',
                section='fixture', verified_by='test', verified_at='2026-01-01',
                detector_model_sha256=sha256(settings.model), validation_report='validation.json')
    settings.rules = tmp_path/'rules.json'
    settings.rules.write_text(json.dumps({'rules': [rule]}))
    result = dict(verdict='violation', violations=[dict(type='Fixture', event_type='fixture-event', description='Observed fixture')])
    assert attach_verified_sources(result, settings)['violations'][0]['rule_references']
    result['violations'][0]['type'] = 'Признаки возможного нарушения'
    assert 'rule_references' not in attach_verified_sources(result, settings)['violations'][0]
    result['violations'][0]['type'] = 'Fixture'
    settings.model.write_bytes(b'different-model')
    assert 'rule_references' not in attach_verified_sources(result, settings)['violations'][0]
