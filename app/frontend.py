"""Streamlit client talks only to a configured loopback FastAPI address."""
import base64
import os
from urllib.parse import urlparse

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title='ForteVision', layout='wide')
st.title('ForteVision')
st.caption('Локальная проверка кассовых видео. Экспериментальная бинарная модель.')
st.info('Тип события требует ручной проверки. Юридическое обоснование не реализовано. Отрицательный результат не гарантирует отсутствия нарушения.')
url = os.getenv('FORTEVISION_API_URL', 'http://127.0.0.1:8000').rstrip('/')
parsed = urlparse(url)
if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost', '::1'):
    st.error('Разрешен только локальный HTTP API на loopback-адресе.')
    st.stop()
api_key = os.getenv('FORTEVISION_API_KEY', '')
session = requests.Session()
session.trust_env = False  # never send local videos through environment proxies
headers = {'X-API-Key': api_key}
video = st.file_uploader('Видео .mov', type=['mov'])
if st.button('Анализировать', disabled=video is None):
    try:
        response = session.post(url+'/api/v1/analyze', files={'file': (video.name, video.getvalue(), 'video/quicktime')}, headers=headers, timeout=60, allow_redirects=False)
        if response.status_code != 202:
            raise RuntimeError(response.json().get('detail', response.text))
        st.session_state['job_id'] = response.json()['job_id']
        st.session_state.pop('result', None)
    except (requests.RequestException, ValueError, RuntimeError) as exc:
        st.error(str(exc))


@st.fragment(run_every=2)
def poll():
    job_id = st.session_state.get('job_id')
    if not job_id:
        return
    try:
        if 'result' not in st.session_state:
            response = session.get(url+f'/api/v1/jobs/{job_id}', headers=headers, timeout=10, allow_redirects=False)
            if response.status_code != 200:
                raise RuntimeError(response.json().get('detail', response.text))
            result = response.json()
        else:
            result = st.session_state['result']
        if result['status'] == 'processing':
            st.status('processing: в очереди или выполняется анализ…', state='running')
            return
        st.session_state['result'] = result
        if result['status'] == 'failed':
            st.error(result['error'])
            return
        st.subheader(result['verdict'])
        if 'elapsed_seconds' in result:
            st.caption(f"Обработка: {result['elapsed_seconds']:.2f} с")
        for event in result['violations']:
            st.subheader(f"{event['type']} · {event['timecode_sec']:.2f} с")
            st.write(event['description'])
            for column, frame in zip(st.columns(len(event['evidence_frames'])), event['evidence_frames']):
                column.image(base64.b64decode(frame['image_base64'], validate=True), caption=f"{frame['timestamp_sec']:.2f} с")
        st.download_button('Скачать JSON', data=__import__('json').dumps(result, ensure_ascii=False, indent=2), file_name='fortevision-result.json', mime='application/json')
    except (requests.RequestException, ValueError, RuntimeError) as exc:
        st.error(str(exc))


poll()
