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
        if parsed.scheme == 'https' and parsed.hostname in ('adilet.zan.kz', 'old.adilet.zan.kz'):
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
    markup('<div class="sidebar-note"><span class="privacy-dot"></span> Локальная обработка<br><small>Видео и кадры остаются на этом компьютере. Без распознавания лиц.</small></div>')
    st.divider()
    st.caption('FORTEVISION / DATATHON MVP')
    st.caption('Экспериментальная модель. Решение принимает специалист.')

markup(f'<div class="topline"><span>РАБОЧЕЕ ПРОСТРАНСТВО <b>/ {escape(page)}</b></span><span class="local-tag">● LOCAL MODE</span></div>')


@st.fragment(run_every=10)
def service_status():
    try:
        response = session.get(url + '/health', timeout=2, allow_redirects=False)
        response.raise_for_status()
        st.caption('● Сервис доступен · обработка на этом компьютере')
    except requests.RequestException:
        st.warning('API недоступен. Запустите локальный сервер и повторите запрос.')


def render_result(result):
    if result['status'] == 'failed':
        st.error('Не удалось завершить анализ')
        st.write(result['error'])
        st.caption('Результат не сформирован. Проверьте файл и доступность локальной модели.')
        return
    events = result['violations']
    flagged = result['verdict'] == 'violation'
    title = 'Есть фрагменты для проверки' if flagged else 'Подозрительные фрагменты не найдены'
    subtitle = ('Модель выделила визуальные признаки. Просмотрите кадры и контекст операции.' if flagged
                else 'Оценки окон ниже порога модели. Это не гарантирует отсутствия нарушения.')
    markup(f'<div class="result-banner {"flagged" if flagged else "clear"}"><span class="eyebrow">АНАЛИЗ ЗАВЕРШЁН</span><h3>{title}</h3><p>{subtitle}</p></div>')
    a, b, c = st.columns(3)
    a.metric('Фрагменты', len(events))
    elapsed = result.get('elapsed_seconds')
    b.metric('Время обработки', f'{elapsed:.1f} с' if isinstance(elapsed, (int, float)) else '—')
    c.metric('Кадры для проверки', sum(len(e['evidence_frames']) for e in events))
    st.caption('Текущая модель определяет общие признаки; тип события и применимость нормы проверяет человек.')
    for index, event in enumerate(events, 1):
        with st.container(border=True):
            markup(f'<div class="event-heading"><span class="event-number">{index:02d}</span><span class="time-pill">{timecode(event["timecode_sec"])}</span></div>')
            st.subheader(event['type'])
            st.write(event['description'])
            for column, frame in zip(st.columns(len(event['evidence_frames'])), event['evidence_frames']):
                column.image(base64.b64decode(frame['image_base64'], validate=True),
                             caption=f'Кадр · {timecode(frame["timestamp_sec"])}', use_container_width=True)
    st.download_button('Скачать результат · JSON', data=json.dumps(result, ensure_ascii=False, indent=2),
                       file_name='fortevision-result.json', mime='application/json', use_container_width=True)
    with st.expander('Технические сведения'):
        st.write(f'ID задания: {result["job_id"]}')
        st.write(f'Ответ модели: {result["verdict"]}')
        st.caption('Время обработки не включает загрузку файла и ожидание в очереди.')


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
            st.caption('Извлечение кадров → оценка временных окон → подготовка кадров для проверки. Модель не обучается.')
            return
        st.session_state['result'] = result
        render_result(result)
    except (requests.RequestException, ValueError, RuntimeError, KeyError, TypeError) as exc:
        st.error(f'Не удалось получить результат: {exc}')
        if st.button('Повторить получение результата', key='retry_result'):
            st.session_state.pop('result', None)
            st.rerun(scope='fragment')


def analysis_page():
    markup('<div class="hero"><div class="hero-copy"><span class="eyebrow">FORTEVISION · ВИДЕОАНАЛИТИКА</span><h1>Внимание к каждой<br><em>операции.</em></h1><p>От видеозаписи — к фрагментам, которые требуют внимания специалиста.</p><div class="hero-tags"><span>Локальный AI</span><span>Точные таймкоды</span><span>Кадры из видео</span></div></div><div class="hero-art" aria-hidden="true"><div class="art-orbit"></div><div class="art-frame"><div class="art-bars"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i></div><div class="art-line"></div><span>FRAME / INSIGHT</span></div><div class="art-chip">◈ <b>ForteVision</b><small>В фокусе — детали</small></div></div></div>')
    service_status()
    left, right = st.columns([0.85, 1.4], gap='large')
    with left:
        markup('<div class="section-heading"><span>01</span><h2>Видеозапись</h2></div>')
        with st.container(border=True):
            st.markdown('**Добавьте запись операции**')
            st.caption('MOV · до 256 МБ · один файл за анализ')
            video = st.file_uploader('Выберите видео .mov', type=['mov'], label_visibility='collapsed')
            if video:
                st.caption(f'{video.name} · {video.size / 1024**2:.1f} МБ')
                with st.expander('Предпросмотр записи'):
                    st.video(video, format='video/quicktime')
                    st.caption('Если браузер не поддерживает кодек MOV, анализ всё равно доступен.')
            if st.button('Начать анализ →', type='primary', disabled=video is None, use_container_width=True):
                try:
                    with st.spinner('Передаём видео локальному сервису…'):
                        response = session.post(url + '/api/v1/analyze',
                            files={'file': (video.name, video.getvalue(), 'video/quicktime')},
                            headers=headers, timeout=60, allow_redirects=False)
                    if response.status_code != 202:
                        raise RuntimeError(response.json().get('detail', response.text))
                    st.session_state['job_id'] = response.json()['job_id']
                    st.session_state['analyzed_filename'] = video.name
                    st.session_state.pop('result', None)
                except (requests.RequestException, ValueError, RuntimeError, KeyError) as exc:
                    st.error(f'Не удалось отправить видео: {exc}')
        markup('<div class="note-card"><span>БЕЗ ИЗМЕНЕНИЯ МОДЕЛИ</span><h4>Загружайте. Проверяйте. Повторяйте.</h4><p>Анализ использует готовую модель. Ваши загрузки не запускают обучение и не меняют её веса.</p></div>')
        with st.expander('Что важно знать о результате'):
            st.write('Это экспериментальная бинарная модель. В проведённой проверке все четыре нормальных test-видео получили ложную тревогу. Каждый результат требует ручной проверки.')
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
    st.info('Справочник для специалиста. Сценарии из research ещё не распознаются текущей моделью. Совпадение с нормой и вывод о нарушении автоматически не устанавливаются.')
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
