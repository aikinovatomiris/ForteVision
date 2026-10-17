import json
import queue
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from .config import ROOT
from .schemas import DoneJob


class QueueFullError(RuntimeError):
    pass


def subprocess_runner(path: Path, settings):
    output = path.with_suffix('.result.json')
    started = time.perf_counter()
    try:
        process = subprocess.run(
            [sys.executable, '-m', 'ml.predict', '--video', str(path), '--model', str(settings.model),
             '--weights', str(settings.weights), '--device', settings.device,
             '--budget', str(max(1, settings.timeout_seconds-10)), '--output', str(output)],
            cwd=ROOT, capture_output=True, text=True, timeout=settings.timeout_seconds,
        )
        if process.returncode:
            try:
                error = json.loads(process.stderr.strip().splitlines()[-1])['error']
            except (ValueError, KeyError, IndexError):
                error = 'Ошибка процесса анализа/декодера; проверьте окружение и локальные веса'
            raise RuntimeError(error)
        if not output.is_file() or output.stat().st_size > 16*1024*1024:
            raise RuntimeError('Отсутствует результат или evidence превышает 16 MB; сократите видео')
        result = json.loads(output.read_text(encoding='utf-8'))
        result['elapsed_seconds'] = time.perf_counter()-started
        result['within_five_minutes'] = result['elapsed_seconds'] < 300
        return result
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f'Лимит {settings.timeout_seconds} секунд превышен; процесс остановлен') from exc
    finally:
        output.unlink(missing_ok=True)
        output.with_suffix(output.suffix+'.tmp').unlink(missing_ok=True)


class JobStore:
    """Single worker, bounded queue and bounded/expiring in-memory results."""
    def __init__(self, settings, runner=None):
        self.settings = settings
        self.runner = runner or (lambda path: subprocess_runner(path, settings))
        self.queue = queue.Queue(maxsize=settings.queue_size)
        self.jobs, self.completed = {}, {}
        self.lock = threading.Lock()
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self._work, name='fortevision-worker', daemon=True)

    def start(self):
        self.thread.start()

    def _prune(self):
        expired = [k for k, t in self.completed.items() if time.monotonic()-t > self.settings.ttl_seconds]
        for key in expired:
            self.jobs.pop(key, None)
            self.completed.pop(key, None)

    def submit(self, path):
        with self.lock:
            self._prune()
            if self.stopping.is_set() or len(self.jobs) >= self.settings.max_jobs:
                raise QueueFullError('Хранилище jobs заполнено; повторите после TTL')
            job_id = str(uuid.uuid4())
            self.jobs[job_id] = dict(job_id=job_id, status='processing')
            try:
                self.queue.put_nowait((job_id, path))
            except queue.Full as exc:
                self.jobs.pop(job_id)
                raise QueueFullError('Очередь заполнена; повторите позже') from exc
            return job_id

    def get(self, job_id):
        with self.lock:
            self._prune()
            job = self.jobs.get(job_id)
            return dict(job) if job else None

    def _work(self):
        while not self.stopping.is_set():
            try:
                job_id, path = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                result = self.runner(path)
                completed = DoneJob(job_id=job_id, status='done', **result).model_dump(exclude_none=True)
            except Exception as exc:
                completed = dict(job_id=job_id, status='failed', error=str(exc)[:2000])
            finally:
                path.unlink(missing_ok=True)
            with self.lock:
                self.jobs[job_id] = completed
                self.completed[job_id] = time.monotonic()
            self.queue.task_done()

    def stop(self):
        self.stopping.set()
        self.thread.join(timeout=self.settings.timeout_seconds+5)
        while True:
            try:
                job_id, path = self.queue.get_nowait()
            except queue.Empty:
                break
            path.unlink(missing_ok=True)
            self.queue.task_done()
            with self.lock:
                self.jobs[job_id] = dict(job_id=job_id, status='failed', error='Сервис остановлен до начала обработки')
