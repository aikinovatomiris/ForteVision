import asyncio
import hmac
import time

from fastapi import HTTPException
from starlette.datastructures import Headers
from starlette.responses import JSONResponse


class RequestGuard:
    """Authenticate before multipart parsing; bound even chunked request bodies."""
    def __init__(self, app, settings):
        self.app, self.settings = app, settings
        self.upload_slots = asyncio.Semaphore(settings.queue_size+1)

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        headers = Headers(scope=scope)
        is_upload = scope['path'] == '/api/v1/analyze' and scope['method'] == 'POST'
        if is_upload:
            scope['fortevision_received_at'] = time.monotonic()
        if scope['path'].startswith('/api/'):
            if not self.settings.api_key or self.settings.api_key.startswith('replace-with-'):
                return await JSONResponse({'detail': 'Настройте FORTEVISION_API_KEY в .env'}, status_code=503)(scope, receive, send)
            supplied = headers.get('x-api-key', '')
            if not hmac.compare_digest(supplied.encode(), self.settings.api_key.encode()):
                return await JSONResponse({'detail': 'Неверный X-API-Key'}, status_code=401)(scope, receive, send)
        limit = self.settings.max_upload_bytes+1024*1024  # multipart overhead
        try:
            length = int(headers.get('content-length', '0'))
        except ValueError:
            return await JSONResponse({'detail': 'Некорректный Content-Length'}, status_code=400)(scope, receive, send)
        if length > limit:
            return await JSONResponse({'detail': 'Файл превышает лимит размера'}, status_code=413)(scope, receive, send)
        received = 0
        async def limited_receive():
            nonlocal received
            if is_upload:
                remaining = self.settings.timeout_seconds-(time.monotonic()-scope['fortevision_received_at'])
                if remaining <= 0:
                    raise HTTPException(408, 'Срок загрузки истёк')
                try:
                    message = await asyncio.wait_for(receive(), timeout=remaining)
                except asyncio.TimeoutError as exc:
                    raise HTTPException(408, 'Срок загрузки истёк') from exc
            else:
                message = await receive()
            received += len(message.get('body', b''))
            if received > limit:
                raise HTTPException(413, 'Файл превышает лимит размера')
            return message
        if is_upload:
            if self.upload_slots.locked():
                return await JSONResponse({'detail': 'Слишком много загрузок'}, status_code=429)(scope, receive, send)
            await self.upload_slots.acquire()
        try:
            await self.app(scope, limited_receive, send)
        finally:
            if is_upload:
                self.upload_slots.release()
