"""One worker, persisted results and an end-to-end deadline; no training."""
from dataclasses import replace
import json
import queue
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from .config import ROOT
from .schemas import DoneJob
from .rule_links import attach_verified_sources
from .runtime_lock import RuntimeLock


class QueueFullError(RuntimeError):
    pass


class JobExpiredError(RuntimeError):
    pass


def subprocess_runner(path: Path, settings, timeout_seconds=None):
    output = path.with_suffix('.result.json')
    remaining = settings.timeout_seconds if timeout_seconds is None else timeout_seconds
    if remaining <= 0:
        raise JobExpiredError('Общий срок задания истёк до начала анализа')
    try:
        process = subprocess.run(
            [sys.executable, '-m', 'ml.predict', '--video', str(path), '--model', str(settings.model),
             '--backend', settings.backend, '--weights', str(settings.weights), '--device', settings.device,
             '--budget', str(max(.01, remaining-2)), '--output', str(output)],
            cwd=ROOT, capture_output=True, text=True, timeout=remaining,
        )
        if process.returncode:
            try:
                error = json.loads(process.stderr.strip().splitlines()[-1])['error']
            except (ValueError, KeyError, IndexError):
                error = 'Ошибка процесса анализа; проверьте окружение и локальные веса'
            raise RuntimeError(error)
        if not output.is_file() or output.stat().st_size > 16*1024*1024:
            raise RuntimeError('Отсутствует результат или evidence превышает 16 MB')
        return json.loads(output.read_text(encoding='utf-8'))
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError('Общий срок задания превышен; процесс остановлен') from exc
    finally:
        output.unlink(missing_ok=True)
        output.with_suffix(output.suffix+'.tmp').unlink(missing_ok=True)


class JobStore:
    """SQLite survives restarts. Only one API process may own a store."""
    def __init__(self, settings, runner=None):
        self.settings, self.runner = settings, runner
        self.queue = queue.Queue(maxsize=settings.queue_size)
        self.jobs, self.completed, self.pending = {}, {}, {}
        self.backends = {}
        self.lock = threading.RLock()
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self._work, name='fortevision-worker', daemon=True)
        self.sweeper = threading.Thread(target=self._sweep, name='fortevision-deadlines', daemon=True)
        self.db = None

    def start(self):
        self.settings.job_db.parent.mkdir(parents=True, exist_ok=True)
        self.lease = RuntimeLock(self.settings.job_db.with_suffix('.lock'))
        self.db = sqlite3.connect(self.settings.job_db, check_same_thread=False)
        self.db.execute('PRAGMA secure_delete=ON')
        self.db.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, payload TEXT NOT NULL, completed REAL)')
        for key, payload, completed in self.db.execute('SELECT id, payload, completed FROM jobs').fetchall():
            job = json.loads(payload)
            if job['status'] == 'processing':
                job = dict(job_id=key, status='failed', error='Сервис перезапущен; загрузите видео повторно')
                completed = time.time()
            self.jobs[key] = job
            self.completed[key] = completed
            self._persist(key)
        self._prune()
        self.thread.start()
        self.sweeper.start()

    def _persist(self, key):
        if self.db is not None:
            self.db.execute('INSERT OR REPLACE INTO jobs VALUES (?,?,?)',
                            (key, json.dumps(self.jobs[key], ensure_ascii=False), self.completed.get(key)))
            self.db.commit()

    def _remove(self, key):
        self.jobs.pop(key, None)
        self.completed.pop(key, None)
        if self.db is not None:
            self.db.execute('DELETE FROM jobs WHERE id=?', (key,))
            self.db.commit()

    def _prune(self):
        for key, ended in list(self.completed.items()):
            if time.time()-ended > self.settings.ttl_seconds:
                self._remove(key)

    def _finish(self, key, job):
        # A late result must never overwrite a timeout already published by the watchdog.
        if key in self.jobs and self.jobs[key]['status'] == 'processing':
            self.jobs[key] = job
            self.completed[key] = time.time()
            self._persist(key)

    def submit(self, path, received_at=None, backend=None):
        if backend is not None and backend not in ('classifier','local_vlm'):
            raise ValueError('Unknown backend')
        received_at = time.monotonic() if received_at is None else received_at
        with self.lock:
            self._prune()
            if time.monotonic() >= received_at+self.settings.timeout_seconds:
                raise JobExpiredError('Срок задания исчерпан при загрузке файла')
            if self.stopping.is_set() or self.queue.full():
                raise QueueFullError('Очередь заполнена; повторите позже')
            # Evict oldest terminal results instead of refusing all new work for an hour.
            while len(self.jobs) >= self.settings.max_jobs and self.completed:
                self._remove(min(self.completed, key=self.completed.get))
            if len(self.jobs) >= self.settings.max_jobs:
                raise QueueFullError('Все доступные места заняты активными заданиями')
            key = str(uuid.uuid4())
            self.jobs[key] = dict(job_id=key, status='processing')
            self.pending[key] = (path, received_at, time.monotonic())
            self.backends[key] = backend or self.settings.backend
            self._persist(key)
            self.queue.put_nowait(key)
            return key

    def get(self, job_id):
        with self.lock:
            self._prune()
            job = self.jobs.get(job_id)
            return dict(job) if job else None

    def delete(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                return False
            if self.jobs[job_id]['status'] == 'processing':
                raise ValueError('Дождитесь завершения задания перед удалением')
            self._remove(job_id)
            return True

    def _sweep(self):
        while not self.stopping.wait(.1):
            with self.lock:
                self._prune()
                for key, (_, received, _) in list(self.pending.items()):
                    if time.monotonic() >= received+self.settings.timeout_seconds:
                        self._finish(key, dict(job_id=key, status='failed', error='Общий срок задания истёк (загрузка + очередь + анализ)'))

    def _work(self):
        while not self.stopping.is_set():
            try:
                key = self.queue.get(timeout=.1)
            except queue.Empty:
                continue
            with self.lock:
                path, received, accepted = self.pending[key]
                active = self.jobs.get(key, {}).get('status') == 'processing'
            started = time.monotonic()
            try:
                remaining = received+self.settings.timeout_seconds-started
                if not active or remaining <= 0:
                    raise JobExpiredError('Срок задания истёк в очереди')
                result = self.runner(path) if self.runner else subprocess_runner(path, replace(self.settings, backend=self.backends[key]), remaining)
                result = attach_verified_sources(result, self.settings)
                elapsed = time.monotonic()-received
                if elapsed >= self.settings.timeout_seconds:
                    raise JobExpiredError('Общий срок задания истёк')
                result.update(elapsed_seconds=elapsed, within_five_minutes=elapsed < 300,
                              queue_seconds=started-accepted, processing_seconds=time.monotonic()-started)
                completed = DoneJob(job_id=key, status='done', **result).model_dump(exclude_none=True)
            except Exception as exc:
                completed = dict(job_id=key, status='failed', error=str(exc)[:2000])
            finally:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    # Recovery cleanup is documented; a cleanup error must not kill the worker.
                    pass
            with self.lock:
                self._finish(key, completed)
                self.pending.pop(key, None)
                self.backends.pop(key, None)
            self.queue.task_done()

    def stop(self):
        self.stopping.set()
        self.sweeper.join(timeout=1)
        self.thread.join(timeout=self.settings.timeout_seconds+3)
        with self.lock:
            while True:
                try:
                    key = self.queue.get_nowait()
                except queue.Empty:
                    break
                path, _, _ = self.pending.pop(key)
                self.backends.pop(key, None)
                path.unlink(missing_ok=True)
                self.queue.task_done()
                self._finish(key, dict(job_id=key, status='failed', error='Сервис остановлен до начала анализа'))
            if self.db and not self.thread.is_alive():
                self.db.close()
                self.db = None
                self.lease.close()
