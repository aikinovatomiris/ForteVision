# ForteVision

Локальный MVP анализа кассовых видео: **ResNet18 embeddings → temporal windows → StandardScaler + LogisticRegression → события → настоящие JPEG evidence**. Распознавания лиц нет. Облачные AI API не используются. Типы нарушений не размечены, поэтому единственная категория — «Признаки возможного нарушения».

**Текущее качество недостаточно для практического контроля:** на 8 отложенных test-видео precision 0.50, recall 1.00, F1 0.667; все 4 отрицательных видео получили ложную тревогу. Порог выбран только на validation. Модель и сервис реально запущены, но качество детектора требует дальнейшей работы. См. [VALIDATION.md](VALIDATION.md) и `artifacts/test_metrics.json`.

## Архитектура

```text
fortevision/
├── app/
│   ├── api.py             # FastAPI, multipart upload, endpoints
│   ├── schemas.py         # Контракт жюри, processing/done/failed
│   ├── config.py          # Environment settings
│   ├── security.py        # API key, лимит тела запроса
│   ├── jobs.py            # Ограниченная очередь, worker, cleanup, TTL
│   └── frontend.py        # Streamlit, только локальный API
├── ml/
│   ├── dataset.py         # Excel, Unicode X/Х, manifest, SHA-256
│   ├── preprocessing.py   # OpenCV, реальные timestamps, temporal pooling
│   ├── detection.py       # ResNet18, offline weights, CPU/CUDA, batches
│   ├── temporal.py        # Метки окон, weak negatives, объединение событий
│   ├── training.py        # Train, validation threshold tuning, test evaluation
│   ├── pipeline.py        # Главная функция анализа и загрузка joblib
│   ├── predict.py         # CLI/изолированный процесс inference
│   ├── evidence.py        # JPEG + base64 с timestamp декодера
│   └── rules.py           # Только проверенные типы и нормативные источники
├── scripts/               # inspect_dataset, contact_sheets, prepare_weights,
│                          # license_inventory
├── tests/                 # dataset, pipeline, API; без скачивания весов
├── configs/rules.json      # Пустой каталог нормативов
├── data/raw/               # open_labels.xlsx, вне Git
├── data/videos/            # .mov, вне Git
├── models/                 # .pth, .joblib, metadata, вне Git
├── artifacts/              # Manifest и отчеты, вне Git; создается командами
├── research/contact_sheets/# Локальные кадры, вне Git; создается командой
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── .streamlit/config.toml
├── Dockerfile
└── LICENSES.md
```

## Окружение Windows

Целевая версия — **Python 3.11 x64**. Зафиксированная пара `torch 2.6 / torchvision 0.21` также поддерживает Python 3.12; Python 3.13/3.14 для этой конфигурации не использовать. Команды запускать из корня проекта.

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& .\.venv\Scripts\Activate.ps1
python --version
python -m pip install -r requirements-dev.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Если `.venv` уже существует, проверьте `python --version`. В подготовленной рабочей папке доступный интерпретатор был Python 3.12.14; это не подтверждение проверки на 3.11. Для 3.11 создайте новое окружение установленным `py -3.11`, не смешивайте библиотеки разных Python. Активация необязательна: можно вызывать `.\.venv\Scripts\python.exe` напрямую.

В новой установке замените `FORTEVISION_API_KEY` в `.env` случайной строкой (например, результатом `python -c "import secrets; print(secrets.token_urlsafe(32))"`). В текущей рабочей папке `.env` уже создан с локально сгенерированным ключом; не перезаписывайте его шаблоном. Placeholder и пустой ключ не разрешают запросы к `/api/`. Не добавляйте `.env` в Git. Для CUDA установите совместимую пару torch/torchvision под ваш драйвер по официальным инструкциям PyTorch и задайте `FORTEVISION_DEVICE=cuda` для сервиса, а CLI-командам обучения/анализа передайте `--device cuda`; CPU — режим по умолчанию.

## Данные и проверка

```powershell
python -m scripts.inspect_dataset
python -m scripts.contact_sheets --video-id X-1
python -m scripts.contact_sheets
```

Excel: `data/raw/open_labels.xlsx`. Колонки: `наименование файла`, `таймстемп`, `нарушение / без нарушения`. Поддерживаются русские метки и 0/1; timestamps числом секунд, `MM:SS`, `HH:MM:SS`, Excel time, `22 секунда`, `1минута 8 секунда`. Число без Excel time-формата трактуется как секунды.

NFKC-нормализация и X/Х используются для сопоставления, исходные имена не меняются. Неоднозначные имена, дубликаты и противоречивые метки запрещены. `inspect_dataset` сохраняет `artifacts/dataset_inspection.json`: число положительных/отрицательных записей, все найденные ошибки, пропущенные и неразмеченные видео, timestamp, длительность, FPS и разрешение. Это проверка метаданных, не полный тест декодирования и не интерпретация содержания.

`contact_sheets` создает JPEG-сетки 3×3 около размеченного timestamp и JSON с фактическими временами кадров в `research/contact_sheets/`. Отрицательные видео без timestamp пропускаются. Кадры остаются локально. Просмотр holdout/test перед выбором модели может влиять на эксперимент: для разработки используйте тренировочные видео.

## Веса до запуска offline inference

```powershell
python -m scripts.prepare_weights
```

Это единственная команда скачивания модели. Она получает официальный `resnet18-f37072fd.pth`, проверяет SHA-256 prefix из официального имени и сохраняет полный digest/provenance рядом. Веса можно заранее перенести с другой машины в `models/`. Условия использования предобученных весов рассмотрены отдельно в [LICENSES.md](LICENSES.md); коммерческая пригодность ImageNet-весов не объявляется автоматически подтвержденной лицензией кода torchvision.

Inference создает `resnet18(weights=None)`, читает **только локальный** state_dict с `weights_only=True`, удаляет `fc`, применяет `ResNet18_Weights.IMAGENET1K_V1.transforms()`. `eval()`, `torch.no_grad()` и batching включены. Скачивания через `torch.hub` в inference нет. При отсутствии или повреждении весов job завершается `failed`.

## Обучение и оценка

```powershell
python -m ml.training prepare
python -m ml.training train --fps 2 --window 4 --stride 2 --batch-size 16
python -m ml.training evaluate
```

`prepare` сначала объединяет точные SHA-256 дубликаты в canonical video с `aliases` и всеми размеченными `timestamps`; конфликтующие бинарные метки требуют исправления человеком. Затем сохраняет `artifacts/manifest.json`, seed=42: приблизительно 60% train, 20% validation, 20% test, строго по уникальному содержимому/video ID, стратифицированно. Существующий manifest не перезаписывается. Неразмеченные файлы не включаются. В текущих данных 42 строки соответствуют 40 уникальным видео: Х-4/Х-47 совпадают (события 57 и 68 с), Х-5/Х-48 совпадают. Получается 24/8/8 уникальных видео; оба события сохраняются в одной выборке.

**Официальный test дататона никогда не должен попадать в train/validation.** Если такие файлы входят в общий Excel, до обучения явно назначьте их canonical группе `test` или `excluded` в manifest. Если тестовым оказался alias, исключается из обучения вся группа идентичного содержимого. Автоматическое разделение предназначено для размеченного development-набора, а не для пересмотра официального split. Внешний test храните отдельно и используйте только для evaluation/inference. Без IDs сессий/камер нельзя автоматически исключить связанность разных записей одной операции: для этого нужен групповой split; перекодированные дубликаты SHA-256 не выявляет.

Кадры с configurable FPS преобразуются в embeddings по 512 признаков. Mean + max pooling дает 1024 признака на overlapping окно. FPS, длина окна, шаг и batch size сохраняются в metadata и повторно используются при inference. Положительные окна пересекают область `timestamp ± positive-radius` (по умолчанию 1 секунда). Остальные окна положительного видео неизвестны.

Опционально:

```powershell
python -m ml.training train --weak-negatives --exclusion-zone 6 --positive-radius 1 --model models/with_weak.joblib
```

Weak negatives допускаются только когда **все окно** находится вне exclusion zones всех известных событий данного содержимого. Они явно помечены, получают вес 0.25 и не смешиваются с подтвержденными окнами в основной метрике качества. По умолчанию отключены: другие события могут быть не размечены. Отрицательные видео дают отрицательные окна.

Обучается sklearn `Pipeline(StandardScaler, LogisticRegression(class_weight="balanced", C=0.1))`. Scaler и классификатор видят только train. Порог выбирается только на validation по video-level F1; при равенстве выбирается больший порог. Test при обучении не декодируется и не используется для подбора параметров (проверяется только целостность файлов). Сохраняются `models/classifier.joblib` и `models/classifier.json` с конфигурацией, порогом, seed, версиями, хешами и validation metrics. Joblib загружайте только из доверенного локального источника: формат может исполнять код, загрузки моделей через API нет.

`evaluate` отдельно пишет `artifacts/test_metrics.json`: precision, recall, F1, confusion matrix, ROC-AUC для video/window уровней, отдельные weak-negative metrics и ошибка локализации. Video score — максимум по окнам. Ошибка локализации — расстояние от разметки до ближайшего предсказанного peak; считается только для положительных видео с найденными событиями, рядом приводится число пропущенных видео. Если метрика не определена, возвращается `null`/причина, не выдуманное значение. Validation после настройки порога оптимистична; итоговая оценка — на нетронутом test. 42 записи недостаточны для надежных выводов о промышленном качестве.

## API и Streamlit

Рекомендуемый единый запуск (API + UI, без обучения, с блокировкой внешних Python-соединений):

```powershell
python -m scripts.serve
```

Развёртывание и HTTPS: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md). Пакет жюри: [docs/JURY.md](docs/JURY.md). Контракт для ML-разработчика: [docs/ML_INTERFACE.md](docs/ML_INTERFACE.md). Хранение и очистка: [docs/DATA_RETENTION.md](docs/DATA_RETENTION.md). Оставшиеся внешние условия: [docs/NON_ML_STATUS.md](docs/NON_ML_STATUS.md).

Для отдельной отладки без egress guard — первый терминал:

```powershell
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000 --workers 1 --no-access-log
```

Второй терминал (то же окружение):

```powershell
python -m streamlit run app/frontend.py
```

Открыть http://127.0.0.1:8501. Streamlit загружает MOV, показывает `processing`, два вердикта из ТЗ, события, таймкоды, JPEG и ошибки. Он обращается только к loopback API, не использует proxy environment и не следует redirect. Телеметрия Streamlit отключена. Отдельный справочник отображает research; ссылки в конкретном результате появляются только при утверждённом сопоставлении типа с нормой и текущим хешем модели. Для бинарного baseline таких сопоставлений пока нет.

```powershell
curl.exe http://127.0.0.1:8000/health
curl.exe -X POST http://127.0.0.1:8000/api/v1/analyze -H "X-API-Key: YOUR_LOCAL_KEY" -F "file=@data/videos/X-1.mov"
curl.exe http://127.0.0.1:8000/api/v1/jobs/JOB_ID -H "X-API-Key: YOUR_LOCAL_KEY"
```

POST возвращает HTTP 202 и только `{"job_id":"..."}`. GET требует тот же ключ и возвращает `processing`, `done` или `failed`. При успехе:

```json
{
  "job_id": "...",
  "status": "done",
  "verdict": "violation",
  "violations": [{
    "type": "Признаки возможного нарушения",
    "timecode_sec": 22.0,
    "description": "Модель обнаружила подозрительный визуальный фрагмент; тип требует ручной проверки.",
    "evidence_frames": [{"timestamp_sec": 22.0, "image_base64": "<real JPEG base64>"}]
  }]
}
```

Пример показывает структуру, а не результат вашего видео. В реальном ответе base64 содержит JPEG из загруженного видео; API валидирует изображение. `no_violation` имеет пустой `violations`; отсутствующая модель дает `failed`, а не фиктивный отрицательный результат. Необязательные поля `elapsed_seconds` и `within_five_minutes` учитывают загрузку API, очередь и обработку; `queue_seconds` и `processing_seconds` показывают составляющие. Для будущих подтверждённых типов поддерживаются optional `event_type`/`rule_references`; обязательный контракт жюри сохранён. Swagger CDN отключен; схема доступна в `/openapi.json`. `/ready` проверяет ключ и целостность моделей; `/health` — только доступность процесса.

## Ограничения сервиса и безопасность данных

- Один процесс Uvicorn и один worker. Очередь по умолчанию 3 задачи, максимум 20 записей jobs, TTL результатов до 1 часа. `processing` означает ожидание **или** выполнение. При перегрузке 429. SQLite сохраняет результаты между перезапусками; прерванные jobs получают failed. При заполнении удаляется старый завершённый результат, а не блокируются все новые загрузки. После TTL/вытеснения/удаления GET возвращает 404. Uploads — в системном TEMP `fortevision-uploads`; SQLite — рядом в `fortevision-uploads-state`. Пути: FORTEVISION_UPLOAD_DIR/FORTEVISION_JOB_DB. Файлы удаляются после обработки; остатки после аварии очищает `python -m scripts.cleanup_runtime --apply` при остановленном API. Подробности в DATA_RETENTION.md.
- MOV: проверка расширения и сигнатуры контейнера при загрузке; полная проверка декодирования в worker. Ограничение 256 MB, UUID имена, исходное имя не используется как путь. Некорректный формат — 415/422, размер — 413, ключ — 401, ключ не настроен — 503. Защита тела запроса также работает без Content-Length.
- Общий серверный бюджет 295 секунд начинается при получении POST и включает загрузку API и очередь. Worker получает оставшееся время; watchdog публикует failed по deadline, поздний результат не заменяет ошибку. Timeout не считается успешной обработкой. Полный клиентский SLA проверяется `scripts.check_api`; сеть/прокси и первая загрузка браузер → Streamlit не измеряются таймером backend. Успех каждого видео менее чем за 5 минут зависит от оборудования и ML. CLI проверяет deadline, зависший native decoder прерывается через API subprocess timeout.
- Ограничения MVP: длительность до часа, максимум 100 событий, JSON результата до 16 MB. Evidence — 1–3 кадра на событие, уменьшенные до 960 px по длинной стороне. Результаты с JPEG занимают память; это одна из причин ограниченного store.
- Времена кадров берутся из OpenCV `CAP_PROP_POS_MSEC`. Отрицательные preroll-кадры до нулевой отметки пропускаются без сдвига временной шкалы. Неисправная/не возрастающая шкала дает ошибку; не подменяется расчетом `frame/FPS`. Evidence извлекается последовательным чтением: seek работает не для всех MOV. Поддержка VFR и кодеков зависит от backend. Метаданные длительности оценены OpenCV. Peak — timestamp реального sampled кадра в окне с максимальной вероятностью; это не доказательство точного момента нарушения.
- Вероятности LogisticRegression не калиброваны для банка. Смена камер, фона и длительности видео может изменить score; максимум по окнам смещается на длинных видео. Нет распознавания лиц, идентификации сотрудников или классификации типов нарушений. Бинарная метка не является юридическим выводом.
- Видео, contact sheets, evidence, labels и веса сохраняются только локально. Защищайте доступ к диску и `.env`. Текущая папка находится внутри **OneDrive**: приложение не отправляет данные наружу, но синхронизация OneDrive управляется отдельно; для строгой локальности перенесите рабочие данные в несинхронизируемый каталог или отключите синхронизацию для него.

## Offline и Docker

На машине с интернетом подготовьте dependencies и веса:

```powershell
python -m pip download -r requirements-dev.txt -d wheelhouse
python -m scripts.prepare_weights
```

Перенесите wheelhouse подходящих Python/OS, веса и обученный classifier с metadata. Установка без сети: `python -m pip install --no-index --find-links wheelhouse -r requirements-dev.txt`. Inference не скачивает модели и не вызывает облачные сервисы. Отключение сети допустимо после установки. Joblib/torch/sklearn версии должны совпадать с обучением; перенос CPU/CUDA проверять отдельно.

Dockerfile использует Python 3.11; модель и данные не включены в image. Единый UI/API:

```powershell
docker compose config --quiet
docker compose up -d --build
```

Docker build требует сети; подготовленный runtime рассчитан на offline. Compose задаёт контейнерные пути моделей. Публичный HTTPS — дополнительный compose.public.yaml, см. DEPLOYMENT.md. Сам Dockerfile не является свидетельством пройденного Docker-теста.

## Тесты

```powershell
python -m pytest -q
python -m unittest discover -s tests -v
python -m compileall -q app ml scripts tests
```

Полный набор запускается через pytest; unittest запускает только совместимую часть и пропускает pytest-функции. При отсутствии библиотек отдельные группы пропускаются с явной причиной; пропуск не равен успешной проверке. Проверяются labels, X/Х/NFKC, timestamps, video split, duplicate hashes, OpenCV decoding, pooling, evidence JPEG, отсутствие модели/весов, authentication, upload limits, bounded queue, processing/done/failed и JSON schema. В API используются mock модели; веса не скачиваются. Синтетические видео/модели создаются только во временных каталогах unit tests и не участвуют в итоговом обучении.

Результаты фактически выполненных проверок фиксируются в [VALIDATION.md](VALIDATION.md). Метрики ResNet и обещания скорости не публикуются до реального прогона.

## Нормативы и развитие

`configs/rules.json` содержит пустой список. `RuleEngine` принимает только записи с проверенным источником, пунктом, проверяющим и датой. Связь общего бинарного score с нормой запрещена. Юридическое обоснование **не реализовано**.

В Streamlit добавлен отдельный раздел «Нормативные акты»: поиск, фильтр по пунктам, условия наблюдения, ограничения и ссылки на источники из локального `research/rules.json`. Статусы и даты приводятся как сведения предоставленного research, без самостоятельного утверждения об актуальности права. Этот справочник не подключает сценарии к бинарному детектору. Если файл отсутствует или повреждён, интерфейс показывает объяснение. В Docker каталог research исключён из образа; при запуске UI в контейнере подключите его отдельно read-only. HTML-отчёт и ZIP не исполняются и не встраиваются в страницу.

Оформление UI: Forte Magenta `#A31551`, Sky Blue `#61CFFF`; CSS хранится в `app/frontend.css`, тема — в `.streamlit/config.toml`. Внешние шрифты и изображения не загружаются. Проверка страниц, поиска, фильтров и состояний результата: `python -m pytest tests/test_frontend.py -q` (эти дополнительные тесты требуют pytest, unittest их не запускает). Анализ через UI не обучает и не изменяет модель.

Переключение темы: меню `⋮ → Settings → Choose app theme → Light / Dark` (или системная тема). Палитра карточек, текста и панели следует нативному `color-scheme` Streamlit через CSS `light-dark()`; используется современный браузер с поддержкой этой функции. Ширина панели — 276 px вместо 336 px (на 17,9% меньше). Кнопки сворачивания и открытия постоянно доступны в соответствующем состоянии, включая мобильную ширину. После обновления CSS обновите вкладку, при необходимости `Ctrl+F5`.

Для multiclass нужны определенные экспертами типы событий, начало/конец каждого события, повторная разметка и контроль согласия аннотаторов, session/camera IDs, фиксированный test и отдельная проверка precision/recall/localization каждого типа. Затем можно добавить temporal/multilabel classifier отдельно от бинарного детектора и связать только подтвержденные типы с проверенными нормативами. Не переименовывайте текущий score в конкретное действие без этих шагов.

Проверка лицензий: [LICENSES.md](LICENSES.md). `python scripts/license_inventory.py` сохраняет локальный реестр версий и поставляемых notices. До коммерческого распространения проверьте также условия pretrained weights, данных, транзитивных пакетов и бинарных кодеков.
