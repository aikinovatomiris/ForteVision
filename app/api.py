import asyncio
import uuid
from typing import Literal
from dataclasses import replace
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, Security, UploadFile
from fastapi.security import APIKeyHeader

from .config import Settings
from .jobs import JobStore, QueueFullError, JobExpiredError
from .schemas import JobAccepted, JobResponse
from .security import RequestGuard
from .readiness import model_ready


def create_app(settings=None, runner=None):
    settings = settings or Settings()
    store = JobStore(settings, runner)

    @asynccontextmanager
    async def lifespan(app):
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        store.start()
        yield
        await asyncio.to_thread(store.stop)

    app = FastAPI(title='ForteVision', version='0.2.0', docs_url=None, redoc_url=None, lifespan=lifespan)
    app.add_middleware(RequestGuard, settings=settings)
    app.state.jobs = store
    key_scheme = APIKeyHeader(name='X-API-Key', auto_error=False)

    @app.get('/health')
    def health():
        ready = settings.model.is_file() and settings.model.with_suffix('.json').is_file() and settings.weights.is_file()
        return {'status': 'ok', 'model_files_present': ready,
                'note': 'Целостность модели проверяется при анализе',
                'authentication_configured': bool(settings.api_key and not settings.api_key.startswith('replace-with-'))}

    @app.post('/api/v1/analyze', status_code=202, response_model=JobAccepted, dependencies=[Security(key_scheme)])
    async def submit(request: Request, file: UploadFile = File(...), backend: Literal['classifier', 'local_vlm'] | None = None):
        suffix = Path(file.filename or '').suffix.casefold()
        if suffix not in {'.mov', '.mp4', '.m4v'}:
            await file.close()
            raise HTTPException(415, 'Поддерживаются MOV, MP4 и M4V')
        path = settings.upload_dir / f'{uuid.uuid4().hex}{suffix}'
        submitted = False
        try:
            size, header = 0, b''
            with path.open('xb') as destination:
                while chunk := await file.read(1024*1024):
                    size += len(chunk)
                    if size > settings.max_upload_bytes:
                        raise HTTPException(413, 'Файл превышает лимит размера')
                    if not header:
                        header = chunk[:32]
                    await asyncio.to_thread(destination.write, chunk)
            if size < 12 or header[4:8] not in (b'ftyp', b'moov', b'mdat', b'wide', b'free', b'skip'):
                raise HTTPException(422, 'Файл не похож на MOV/MP4/M4V контейнер')
            try:
                job_id = store.submit(path, request.scope.get('fortevision_received_at'), backend=backend)
            except QueueFullError as exc:
                raise HTTPException(429, str(exc), headers={'Retry-After': '5'}) from exc
            except JobExpiredError as exc:
                raise HTTPException(408, str(exc)) from exc
            submitted = True
            return {'job_id': job_id}
        finally:
            await file.close()
            if not submitted:
                path.unlink(missing_ok=True)

    @app.get('/api/v1/jobs/{job_id}', response_model=JobResponse, response_model_exclude_none=True, dependencies=[Security(key_scheme)])
    def get_job(job_id: uuid.UUID):
        job = store.get(str(job_id))
        if job is None:
            raise HTTPException(404, 'Job не найден или истек TTL')
        return job

    @app.get('/ready')
    def ready(backend: Literal['classifier', 'local_vlm'] | None = None):
        from fastapi.responses import JSONResponse
        ready = model_ready(replace(settings, backend=backend) if backend else settings)
        return JSONResponse({'status': 'ready' if ready else 'not_ready'}, status_code=200 if ready else 503)

    @app.delete('/api/v1/jobs/{job_id}', status_code=204, dependencies=[Security(key_scheme)])
    def delete_job(job_id: uuid.UUID):
        try:
            deleted = store.delete(str(job_id))
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        if not deleted:
            raise HTTPException(404, 'Job не найден')

    return app


app = create_app()
