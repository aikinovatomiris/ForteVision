"""Local ForteVision workspace. Presentation only; never trains the model."""
import base64
import html
import json
import os
from pathlib import Path
from urllib.parse import urlparse

import requests
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')
st.set_page_config(page_title='ForteVision · Аналитика операций', page_icon='◈', layout='wide')
st.markdown(f'<style>{(ROOT / "app/frontend.css").read_text(encoding="utf-8")}</style>', unsafe_allow_html=True)


def markup(text):
    st.markdown(text, unsafe_allow_html=True)


def escape(value):
    return html.escape(str(value))


def timecode(seconds):
    minutes, remainder = divmod(float(seconds), 60)
    return f'{int(minutes):02d}:{remainder:04.1f}'


def load_research():
    try:
        data = json.loads((ROOT / 'research/rules.json').read_text(encoding='utf-8-sig'))
        if not isinstance(data, dict) or not isinstance(data.get('rules'), list):
            raise ValueError('Ожидается объект с массивом rules.')
        for rule in data['rules']:
            if not isinstance(rule, dict) or not all(isinstance(rule.get(k), str) for k in
                    ('rule_id', 'event_name_ru', 'source_clause', 'legal_requirement')):
                raise ValueError('В карточке отсутствуют обязательные текстовые поля.')
        return data, None
    except (OSError, ValueError) as exc:
        return None, f'Не удалось прочитать research/rules.json: {exc}'


def source_link(url):
    if isinstance(url, str):
        parsed = urlparse(url)
        if parsed.scheme == 'https' and parsed.hostname in ('adilet.zan.kz', 'old.adilet.zan.kz', 'nationalbank.kz', 'www.nationalbank.kz'):
            st.link_button('Открыть источник ↗', url, use_container_width=True)


url = os.getenv('FORTEVISION_API_URL', 'http://127.0.0.1:8000').rstrip('/')
parsed = urlparse(url)
if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost', '::1'):
    st.error('Разрешён только локальный HTTP API на loopback-адресе.')
    st.stop()
session = requests.Session()
session.trust_env = False
headers = {'X-API-Key': os.getenv('FORTEVISION_API_KEY', '')}
catalog, catalog_error = load_research()

with st.sidebar:
    markup('<div class="brand"><span class="brand-icon">f<span>v</span></span><div>Forte<span>Vision</span><small>VIDEO INTELLIGENCE</small></div></div>')
    markup('<div class="nav-label">РАБОЧЕЕ ПРОСТРАНСТВО</div>')
    page = st.radio('Раздел', ['Анализ видео', 'Нормативные акты'], label_visibility='collapsed')
    markup('<div class="sidebar-note"><span class="privacy-dot"></span> Локальная обработка<br><small>Видео обрабатывается внутри сервиса, без облачных AI API и распознавания лиц.</small></div>')
    st.divider()
    st.caption('FORTEVISION / DATATHON MVP')
    st.caption('Экспериментальная модель. Решение принимает специалист.')

markup(f'<div class="topline"><span>РАБОЧЕЕ ПРОСТРАНСТВО <b>/ {escape(page)}</b></span><span class="local-tag">● LOCAL MODE</span></div>')


@st.fragment(run_every=10)
def service_status():
    try:
        response = session.get(url + '/ready', timeout=5, allow_redirects=False)
        response.raise_for_status()
        st.caption('● Сервис готов · локальная модель проверена')
    except requests.RequestException:
        st.warning('Сервис не готов: проверьте запуск API, ключ и файлы модели.')


def render_result(result):
    if result['status'] == 'failed':
        st.error('Не удалось завершить анализ')
        st.write(result['error'])
        st.caption('Результат не сформирован. Проверьте файл и доступность локальной модели.')
        return
    review = result.get('review_candidates', [])
    events = result['violations'] + review
    flagged = result['verdict'] == 'violation'
    title = 'Обнаружены признаки возможных нарушений' if flagged else 'Нарушение не обнаружено'
    if result.get('analysis_complete') is False and not flagged:
        title = 'Есть непроверенные шаги операции'
    if result.get('review_required'):
        title = 'Нужна проверка выделенных моментов'
    subtitle = ('Модель выделила визуальные признаки. Просмотрите кадры и контекст операции.' if flagged
                else 'Подтверждающей последовательности не найдено. Непроверенные шаги могут остаться; это не гарантирует отсутствия нарушения.')
    markup(f'<div class="result-banner {"flagged" if (flagged or result.get("review_required")) else "clear"}"><span class="eyebrow">АНАЛИЗ ЗАВЕРШЁН</span><h3>{title}</h3><p>{subtitle}</p></div>')
    a, b, c = st.columns(3)
    a.metric('Моменты для просмотра', len(events))
    elapsed = result.get('elapsed_seconds')
    b.metric('Время обработки', f'{elapsed:.1f} с' if isinstance(elapsed, (int, float)) else '—')
    c.metric('Кадры для проверки', sum(len(e['evidence_frames']) for e in events))
    st.caption('Тип и описание — гипотеза модели. Сопоставьте их с последовательностью кадров; применимость нормы проверяет человек.')
    if result.get('workflow_checks'):
        st.subheader('Последовательность операции')
        for check in result['workflow_checks']:
            state = 'Замечено' if check['state'] == 'observed' else 'Не удалось проверить'
            times = ', '.join(timecode(t) for t in check.get('timestamps', []))
            st.write(f"{check['title']} — {state}" + (f" · {times}" if times else ''))
            st.caption(check['note'])
        with st.expander('Наблюдения модели по времени'):
            for observation in result.get('observations', []):
                st.write(f"{timecode(observation['timestamp_sec'])} · {observation['detail']}")
        for limitation in result.get('analysis_limitations', []):
            st.caption(limitation)
    for index, event in enumerate(events, 1):
        with st.container(border=True):
            markup(f'<div class="event-heading"><span class="event-number">{index:02d}</span><span class="time-pill">{timecode(event["timecode_sec"])}</span></div>')
            if event.get('review_reason'):
                st.caption('ПРИОРИТЕТНАЯ ПРОВЕРКА' if event.get('review_priority') == 'high' else 'ТРЕБУЕТ ПРОСМОТРА')
                st.caption(event['review_reason'])
            st.subheader(('Гипотеза: ' if event.get('review_reason') else '') + event['type'])
            st.markdown('**Что могло произойти**')
            st.write(event['description'])
            st.caption('Кадры ниже помогают проверить это предположение; отдельный кадр может не показывать всё действие.')
            start = event.get('context_start_sec')
            start = max(0., event['timecode_sec'] - 5.) if start is None else start
            end = event.get('context_end_sec')
            end = event['timecode_sec'] + 5. if end is None else end
            with st.expander(f'Посмотреть действие в контексте · {timecode(start)}–{timecode(end)}'):
                if st.session_state.get('analyzed_video_bytes') and st.session_state.get('analyzed_video_job_id') == result['job_id']:
                    st.video(st.session_state['analyzed_video_bytes'], format=st.session_state.get('analyzed_video_mime', 'video/mp4'), start_time=start, end_time=end)
                    st.caption('Исходная запись: до и после выделенного события. Если видео не воспроизводится, проверьте этот интервал в проигрывателе.')
                else:
                    st.info('Исходное видео недоступно в этой сессии. Загрузите его и запустите анализ повторно для просмотра контекста.')
                if not event.get('event_type'):
                    st.write('Почему нет конкретного действия: модель не выбрала достаточно уверенно ни один изученный тип. Этот сигнал сам по себе не подтверждает нарушение.')
            for reference in event.get('rule_references') or []:
                st.markdown('**Нормативное основание для ручной проверки**')
                st.write(reference['source_title'])
                st.caption(f'Пункт / раздел: {reference["section"]} · Проверено: {reference["verified_at"]}')
                source_link(reference['source_url'])
                if reference.get('applicability'):
                    st.caption(reference['applicability'])
            for column, frame in zip(st.columns(len(event['evidence_frames'])), event['evidence_frames']):
                column.image(base64.b64decode(frame['image_base64'], validate=True),
                             caption=f'Кадр · {timecode(frame["timestamp_sec"])}', use_container_width=True)
    st.download_button('Скачать результат · JSON', data=json.dumps(result, ensure_ascii=False, indent=2),
                       file_name='fortevision-result.json', mime='application/json', use_container_width=True)
    with st.expander('Технические сведения'):
        st.write(f'ID задания: {result["job_id"]}')
        st.write(f'Статус проверки: {result.get("assessment_status", result["verdict"])}')
        if result.get('review_required'):
            st.caption('Поле verdict сохраняет старый бинарный формат API. no_violation при needs_review не означает чистую запись.')
        st.caption('Общее время включает загрузку в API, очередь и анализ; загрузка из браузера в Streamlit учитывается отдельно.')
        if 'queue_seconds' in result:
            st.write(f'В очереди: {result["queue_seconds"]:.1f} с')
    if st.button('Удалить результат из сервиса', key='delete_result', use_container_width=True):
        try:
            response = session.delete(url + f'/api/v1/jobs/{result["job_id"]}', headers=headers, timeout=10, allow_redirects=False)
            if response.status_code not in (204, 404):
                raise RuntimeError('Не удалось удалить результат')
            for key in ('result', 'job_id', 'analyzed_filename', 'analyzed_video_bytes', 'analyzed_video_mime', 'analyzed_video_job_id'):
                st.session_state.pop(key, None)
            st.rerun()
        except (requests.RequestException, RuntimeError) as exc:
            st.error(str(exc))


@st.fragment(run_every=2)
def poll():
    job_id = st.session_state.get('job_id')
    if not job_id:
        markup('<div class="empty-state"><div class="scan-icon"><i></i><i></i><i></i></div><h3>Здесь появится результат</h3><p>Загрузите запись и запустите анализ.<br>Мы покажем фрагменты, таймкоды и кадры для ручной проверки.</p><span>01 ЗАГРУЗКА <b>→</b> 02 АНАЛИЗ <b>→</b> 03 ПРОВЕРКА</span></div>')
        return
    try:
        result = st.session_state.get('result')
        if result is None:
            response = session.get(url + f'/api/v1/jobs/{job_id}', headers=headers, timeout=10, allow_redirects=False)
            if response.status_code != 200:
                raise RuntimeError(response.json().get('detail', response.text))
            result = response.json()
        if result['status'] == 'processing':
            st.status('Анализируем запись…', state='running')
            st.write('Видео в очереди или обрабатывается. Результат появится автоматически.')
            st.caption('Кадры по времени → наблюдаемые действия → проверка последовательности → фрагменты для проверки. Модель не обучается.')
            return
        st.session_state['result'] = result
        render_result(result)
    except (requests.RequestException, ValueError, RuntimeError, KeyError, TypeError) as exc:
        st.error(f'Не удалось получить результат: {exc}')
        if st.button('Повторить получение результата', key='retry_result'):
            st.session_state.pop('result', None)
            st.rerun(scope='fragment')


def analysis_page():
    st.warning('Экспериментальный анализ. Проверьте действия и таймкоды: модель может пропустить короткий жест, перепутать предметы или не увидеть сейф. Качество на новых ракурсах ещё не подтверждено.')
    markup('<div class="hero"><div class="hero-copy"><span class="eyebrow">FORTEVISION · ВИДЕОАНАЛИТИКА</span><h1>Внимание к каждой<br><em>операции.</em></h1><p>От видеозаписи — к фрагментам, которые требуют внимания специалиста.</p><div class="hero-tags"><span>Локальный AI</span><span>Точные таймкоды</span><span>Кадры из видео</span></div></div><div class="hero-art" aria-hidden="true"><div class="art-orbit"></div><div class="art-frame"><div class="art-bars"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i></div><div class="art-line"></div><span>FRAME / INSIGHT</span></div><div class="art-chip">◈ <b>ForteVision</b><small>В фокусе — детали</small></div></div></div>')
    service_status()
    left, right = st.columns([0.85, 1.4], gap='large')
    with left:
        markup('<div class="section-heading"><span>01</span><h2>Видеозапись</h2></div>')
        with st.container(border=True):
            mode = st.selectbox('Режим анализа', ['Текущая модель', 'Последовательность действий — эксперимент'])
            backend = 'local_vlm' if mode.startswith('Последовательность') else 'classifier'
            if backend == 'local_vlm':
                st.info('Новый локальный анализ шагов. Пока пропускает нарушения на контрольных видео; используйте для проверки наблюдений, не для оценки качества.')
            st.markdown('**Добавьте запись операции**')
            st.caption('MOV / MP4 / M4V · до 256 МБ · один файл за анализ')
            video = st.file_uploader('Выберите видео MOV, MP4 или M4V', type=['mov', 'mp4', 'm4v'], label_visibility='collapsed')
            if video:
                st.caption(f'{video.name} · {video.size / 1024**2:.1f} МБ')
                with st.expander('Предпросмотр записи'):
                    st.video(video, format='video/quicktime' if Path(video.name).suffix.lower()=='.mov' else 'video/mp4')
                    st.caption('Если браузер не воспроизводит кодек, анализ может быть доступен; для просмотра используйте MP4 с H.264.')
            if st.button('Начать анализ →', type='primary', disabled=video is None, use_container_width=True):
                try:
                    with st.spinner('Передаём видео локальному сервису…'):
                        response = session.post(url + '/api/v1/analyze', params={'backend': backend},
                            files={'file': (video.name, video.getvalue(), 'video/quicktime' if Path(video.name).suffix.lower()=='.mov' else 'video/mp4')},
                            headers=headers, timeout=60, allow_redirects=False)
                    if response.status_code != 202:
                        raise RuntimeError(response.json().get('detail', response.text))
                    st.session_state['job_id'] = response.json()['job_id']
                    st.session_state['analyzed_filename'] = video.name
                    st.session_state['analyzed_video_bytes'] = video.getvalue()
                    st.session_state['analyzed_video_mime'] = 'video/quicktime' if Path(video.name).suffix.lower()=='.mov' else 'video/mp4'
                    st.session_state['analyzed_video_job_id'] = st.session_state['job_id']
                    st.session_state.pop('result', None)
                except (requests.RequestException, ValueError, RuntimeError, KeyError) as exc:
                    st.error(f'Не удалось отправить видео: {exc}')
        markup('<div class="note-card"><span>БЕЗ ИЗМЕНЕНИЯ МОДЕЛИ</span><h4>Загружайте. Проверяйте. Повторяйте.</h4><p>Анализ использует готовую модель. Ваши загрузки не запускают обучение и не меняют её веса.</p></div>')
        with st.expander('Что важно знать о результате'):
            st.write('Это экспериментальная система анализа действий и их порядка. Возможны неверные типы, пропуски и ложные тревоги. Каждый результат требует ручной проверки.')
            st.write('Распознавания лиц нет. Юридическое заключение не формируется. Нормативный справочник доступен в отдельном разделе.')
    with right:
        markup('<div class="section-heading"><span>02</span><h2>Результат анализа</h2></div>')
        if st.session_state.get('analyzed_filename'):
            st.caption(f'Запись: {st.session_state["analyzed_filename"]}')
        poll()


def research_page():
    markup('<div class="research-hero"><span class="eyebrow">БАЗА ЗНАНИЙ / КАЗАХСТАН</span><h1>Нормативные акты<span>.</span></h1><p>Источники, условия наблюдения и ограничения — в одном пространстве для ручной проверки.</p></div>')
    if catalog_error:
        st.warning(catalog_error)
        st.info('Добавьте исходный каталог research/rules.json. Сервис анализа работает независимо от справочника.')
        return
    rules = catalog['rules']
    a, b, c = st.columns(3)
    a.metric('Сценариев в справочнике', len(rules))
    b.metric('Нормативных источников', len({r.get('source_url') for r in rules if r.get('source_url')}))
    c.metric('Дата research-каталога', catalog.get('as_of', 'Не указана'))
    st.info('Справочник для специалиста. Модель предполагает типы действий, но их соответствие пунктам этого справочника ещё не проверено. Вывод о нарушении нормы автоматически не устанавливается.')
    search, filter_column = st.columns([3, 1])
    query = search.text_input('Поиск по справочнику', placeholder='Название, пункт или текст требования…')
    selected = filter_column.selectbox('Пункт источника', ['Все пункты'] + sorted({r['source_clause'] for r in rules}))
    matches = [r for r in rules if (selected == 'Все пункты' or r['source_clause'] == selected)
               and query.casefold().strip() in ' '.join(str(r.get(k, '')) for k in
                   ('rule_id', 'event_name_ru', 'source_clause', 'legal_requirement', 'legal_source')).casefold()]
    st.caption(f'Показано {len(matches)} из {len(rules)} сценариев · источник содержания: локальный research/rules.json')
    if not matches:
        st.info('Ничего не найдено. Измените поисковый запрос или выберите все пункты.')
    for rule in matches:
        with st.container(border=True):
            markup(f'<div class="rule-meta"><span class="rule-id">{escape(rule["rule_id"])}</span><span>ПУНКТ {escape(rule["source_clause"])}</span><span class="candidate-tag">СЦЕНАРИЙ ДЛЯ ПРОВЕРКИ</span></div>')
            st.subheader(rule['event_name_ru'])
            st.write(rule['legal_requirement'])
            with st.expander('Условия, источник и ограничения'):
                st.markdown('**Нормативный источник — по данным research**')
                st.write(rule.get('legal_source', 'Название не указано'))
                st.caption(f'Пункт: {rule["source_clause"]}')
                source_link(rule.get('source_url'))
                col1, col2 = st.columns(2, gap='large')
                with col1:
                    st.markdown('**Что должно быть видно**')
                    for item in rule.get('observable_actions', []):
                        st.write('• ' + str(item))
                    for item in rule.get('detection_conditions', {}).get('all_of', []):
                        st.write('• ' + str(item))
                    st.markdown('**Нормальное поведение**')
                    st.write(rule.get('normal_behavior', 'Не описано в каталоге.'))
                with col2:
                    st.markdown('**Когда вывод делать нельзя**')
                    for item in rule.get('detection_conditions', {}).get('suppress_if', []):
                        st.write('• ' + str(item))
                    for item in rule.get('confidence_limitations', []):
                        st.write('• ' + str(item))
                st.markdown('**Какие материалы нужны для проверки**')
                evidence = rule.get('evidence_requirements', {})
                st.write(evidence.get('sequence', 'Не указано.'))
                if evidence.get('continuous_clip_required'):
                    st.caption('По каталогу требуется непрерывный видеофрагмент: отдельных кадров недостаточно.')
                status = rule.get('legal_status', {})
                st.caption(f'Дата проверки, указанная в research: {status.get("as_of", "не указана")}. Статус источника в файле: {status.get("source_status", "не указан")}.')
    with st.expander('Происхождение и актуальность справочника'):
        st.write('Раздел отображает предоставленный локальный каталог, а не результат самостоятельной юридической проверки интерфейсом.')
        st.write(catalog.get('source_currency_warning', 'Ограничения актуальности не указаны в файле.'))
        st.caption('Внешний источник открывается только по нажатию ссылки. Видео и результаты анализа туда не передаются приложением.')
        st.download_button('Скачать research-каталог · JSON', json.dumps(catalog, ensure_ascii=False, indent=2),
                           file_name='fortevision-research-rules.json', mime='application/json')


if page == 'Анализ видео':
    analysis_page()
else:
    research_page()
markup('<div class="footer"><span>ForteVision <b>·</b> Видеоаналитика кассовых операций</span><span>DATATHON MVP / HUMAN REVIEW</span></div>')
