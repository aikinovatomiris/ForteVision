# Развёртывание ForteVision

## Локальный запуск одной командой

Python 3.11/3.12 x64, зависимости `requirements-dev.txt`, подготовленные доверенные модели в `models/`.

```powershell
& .\.venv\Scripts\Activate.ps1
python -m scripts.serve
```

UI: http://127.0.0.1:8501. API: http://127.0.0.1:8000. Остановить через Ctrl+C. Если порты заняты старым запуском, остановить именно старые процессы проекта; новый launcher не завершает чужие процессы. Другие порты: `--api-port 8001 --ui-port 8502`.

В `.env` нужен непустой `FORTEVISION_API_KEY`, не значение из шаблона. Генерация: `python -c "import secrets; print(secrets.token_urlsafe(32))"`; сохраните результат в `.env`, не в Git и не в аргументах команд. Для новой установки копируйте `.env.example`; существующий `.env` не перезаписывайте.

Launcher включает Python egress guard: подключения и DNS к внешним адресам запрещены, loopback доступен UI/API. Это дополнительная защита, не системный firewall и не обещание изоляции стороннего native-кода. Анализ не обучает модель. Для OS-изоляции ограничьте исходящие соединения процесса/контейнера средствами администратора.

`GET /health` — жив ли сервер; `GET /ready` — настроен ли ключ, загружается ли classifier и совпадают ли хеши metadata/весов. `/ready` не проверяет точность ML. Нет модели: 503 readiness, анализ завершается failed; выдуманного отрицательного результата нет.

## Docker: UI и API

```powershell
docker compose config --quiet
docker compose up -d --build
docker compose ps
docker compose logs --tail 100 app
```

Оба сервиса в одном контейнере, поэтому loopback API доступен Streamlit. На машине открыты только loopback-порты 8000/8501. Модели, configs и research подключаются read-only; видео не копируются в образ. SQLite хранится в volume `job_state`; uploads — в tmpfs. Для tmpfs предусмотрено 1 GiB: учитывайте память и число загрузок. Один API worker; второй процесс с той же БД не запускается благодаря OS-lock.

Image build/install требует интернета. После подготовки images и моделей runtime может работать offline. Docker Engine должен быть запущен. Наличие Dockerfile не заменяет проверку build/run на целевой машине. Базовый Python в Docker — 3.11.

## Публичный URL жюри

Нужен ваш сервер с Docker Engine/Compose, домен с DNS на публичный IP, входящие TCP 80/443, достаточная память и разрешение использовать предоставленные видео на этом хосте. Не используйте сторонние AI endpoints, CDN-кеширование ответов/evidence и облачные сервисы анализа. Публичный доступ сам по себе не означает облачный AI: inference остаётся внутри вашего сервиса.

Добавьте в локальный `.env`:

```dotenv
FORTEVISION_DOMAIN=your-real-domain.example
ACME_EMAIL=your-contact@example.com
```

Значения — шаблоны; их необходимо заменить. Затем:

```powershell
docker compose -f compose.yaml -f compose.public.yaml config --quiet
docker compose -f compose.yaml -f compose.public.yaml up -d --build
```

Caddy получает HTTPS-сертификат. `/api/*`, `/health`, `/ready`, `/openapi.json` идут в API; остальные пути и WebSocket — в Streamlit. UI использует внутренний адрес 127.0.0.1:8000; менять его на публичный URL не нужно. API-ключ проверяет backend. Публичный UI не требует авторизации по ТЗ; ограничивайте доступ к демонстрации на уровне своей инфраструктуры, если сервис открыт дольше времени проверки.

Прокси не пишет access logs с запросами, лимит тела 270 MB с запасом под multipart. Порты приложения 8000/8501 остаются только на loopback хоста; наружу публикуются 80/443. Caddy требует исходящего доступа для ACME; Python-процессы работают с egress guard.

Адреса для передачи: `https://ВАШ-ДОМЕН/` и `https://ВАШ-ДОМЕН/api/v1/analyze`. Не заявляйте доступность до реальной проверки с другой машины. Если меняется модель, согласуйте замену с ML-разработчиком, остановите сервис, замените согласованный комплект joblib/metadata/weights и перезапустите.

Конфигурация основана на первичных документах: [Caddy reverse_proxy](https://caddyserver.com/docs/caddyfile/directives/reverse_proxy), [лимит тела запроса](https://caddyserver.com/docs/caddyfile/directives/request_body), [объединение Compose-файлов](https://docs.docker.com/compose/how-tos/multiple-compose-files/merge/).

## Offline-перенос

На машине с интернетом подготовьте wheels под ту же ОС/архитектуру/Python: `python -m pip download -r requirements-dev.txt -d wheelhouse`. В закрытом контуре: `python -m pip install --no-index --find-links wheelhouse -r requirements-dev.txt`.

Перенесите код, согласованные модели, configs, research и notices; не переносите development-видео без необходимости. Для контейнера заранее экспортируйте образ `docker save -o fortevision-image.tar fortevision:local` и загрузите `docker load -i fortevision-image.tar`. При offline-запуске не используйте `--build`/`--pull`; публичный ACME требует сети, в закрытом контуре применяйте выданные администратором сертификаты/прокси.

Проверка без внешних Python-соединений:

```powershell
python -m scripts.verify_offline --video data/videos/X-2.mov
```

Для отдельного OS-level теста inference можно запускать подготовленный контейнер с `--network none`; такая проверка здесь не считается выполненной автоматически. Прямой запуск `uvicorn` вместо `scripts.serve` не включает Python egress guard.
