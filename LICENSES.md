# Зависимости и предобученные веса

Проверены опубликованные лицензии основных библиотек. Это технический реестр, не юридическое заключение. Зафиксированные версии приведены в `requirements.txt`. Лицензия исходного кода библиотеки не автоматически определяет условия всех бинарных компонентов, данных и предобученных весов.

| Компонент | Лицензия кода / первичный источник |
|---|---|
| PyTorch | BSD-style; [LICENSE и сторонние notices](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| torchvision | BSD-3-Clause; [репозиторий и раздел pretrained licenses](https://github.com/pytorch/vision#pre-trained-model-license) |
| OpenCV >=4.5 | Apache-2.0; [официальная страница](https://opencv.org/license/) |
| opencv-python-headless packaging | MIT; [репозиторий](https://github.com/opencv/opencv-python), [third-party licenses](https://github.com/opencv/opencv-python/blob/4.x/LICENSE-3RD-PARTY.txt) |
| NumPy | BSD-3-Clause; [официальный текст](https://numpy.org/doc/stable/license.html) |
| scikit-learn | BSD-3-Clause; [репозиторий](https://github.com/scikit-learn/scikit-learn) |
| pandas | BSD-3-Clause; [LICENSE](https://github.com/pandas-dev/pandas/blob/main/LICENSE) |
| openpyxl | MIT; [метаданные проекта](https://pypi.org/project/openpyxl/) |
| joblib | BSD-3-Clause; [LICENSE](https://github.com/joblib/joblib/blob/main/LICENSE.txt) |
| FastAPI | MIT; [LICENSE](https://github.com/fastapi/fastapi/blob/master/LICENSE) |
| Uvicorn | BSD-3-Clause; [LICENSE](https://github.com/encode/uvicorn/blob/master/LICENSE.md) |
| Streamlit | Apache-2.0; [LICENSE](https://github.com/streamlit/streamlit/blob/develop/LICENSE) |
| requests | Apache-2.0; [LICENSE](https://github.com/psf/requests/blob/main/LICENSE) |
| Pillow | MIT-CMU / HPND; [LICENSE](https://github.com/python-pillow/Pillow/blob/main/LICENSE) |
| python-multipart | Apache-2.0; [LICENSE](https://github.com/Kludex/python-multipart/blob/master/LICENSE.txt) |
| python-dotenv | BSD-3-Clause; [LICENSE](https://github.com/theskumar/python-dotenv/blob/main/LICENSE) |
| pytest / HTTPX | MIT / BSD-3-Clause; [pytest](https://github.com/pytest-dev/pytest/blob/main/LICENSE), [HTTPX](https://github.com/encode/httpx/blob/master/LICENSE.md) |

Эти лицензии кода допускают коммерческое использование при соблюдении условий, включая notices. Но утверждение «весь проект и все веса юридически разрешены для коммерческого применения» здесь не делается.

**ResNet18 IMAGENET1K_V1:** torchvision прямо предупреждает, что pretrained models могут иметь собственные условия, связанные с обучающими данными. Используется официальный файл `https://download.pytorch.org/models/resnet18-f37072fd.pth`; программа проверяет hash prefix и сохраняет полный SHA-256. Коммерческая пригодность этих pretrained weights для банковского внедрения в этом MVP **не подтверждена**. Нужна отдельная проверка условий конкретных весов/данных правообладателем проекта; нельзя выводить разрешение только из BSD лицензии torchvision. См. [официальное предупреждение](https://github.com/pytorch/vision#pre-trained-model-license).

В wheels OpenCV есть FFmpeg и другие сторонние компоненты со своими условиями, включая LGPL; весь wheel нельзя объявлять «только Apache». Проверяйте поставляемые notices, способы распространения/модификации и применимые условия кодеков конкретной сборки. CUDA binary packages также имеют собственные компоненты и условия.

`scripts/license_inventory.py` копирует поставляемые LICENSE/COPYING/NOTICE и метаданные всех установленных пакетов в `artifacts/licenses/`, build information OpenCV — в `artifacts/opencv-build.txt`. Отсутствующие метаданные лицензии не означают разрешение. Полная проверка транзитивных бинарных компонентов требует установленного окружения и анализа сохраненных notices.

Права на банковские видео, разметку и обученный classifier не определяются лицензиями Python-библиотек. Условия дататона и дальнейшего использования исходных данных проверяются отдельно. Research-справочник содержит источники и кандидатные сценарии; runtime-каталог configs/rules.json содержит сопоставления нормативного контекста с отдельными типами гипотез и локальным аудитом источников. Это не подтверждение качества детектора или юридической применимости к конкретному видео; общий бинарный score не получает произвольную нормативную ссылку.

Для сдачи владелец проекта должен зафиксировать условия использования/передачи собственного кода и подтверждение допустимости конкретных pretrained-весов. Добавление backend-инфраструктуры не заменяет это подтверждение. Публичный proxy Caddy добавлен как отдельный контейнер; перед распространением образа включите его LICENSE/NOTICE вместе с notices базового Python/Debian-образа. Коммерческие разрешения не выдаются этим техническим документом.
