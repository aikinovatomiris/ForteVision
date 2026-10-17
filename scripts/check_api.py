"""Jury-style API smoke/load check. Videos go only to the explicitly chosen service."""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

from app.schemas import DoneJob


def check_video(url, key, video):
    session = requests.Session()
    session.trust_env = False
    headers = {'X-API-Key': key}
    started = time.monotonic()
    try:
        with video.open('rb') as stream:
            response = session.post(url+'/api/v1/analyze', files={'file': (video.name, stream, 'video/quicktime')},
                                    headers=headers, timeout=(10, 295), allow_redirects=False)
        if response.status_code != 202:
            raise RuntimeError(f'POST returned HTTP {response.status_code}')
        accepted = response.json()
        if set(accepted) != {'job_id'}:
            raise ValueError('POST must return only job_id')
        statuses = []
        while time.monotonic()-started < 300:
            remaining = 300-(time.monotonic()-started)
            response = session.get(url+'/api/v1/jobs/'+accepted['job_id'], headers=headers,
                                   timeout=max(.1, min(10, remaining)), allow_redirects=False)
            response.raise_for_status()
            result = response.json()
            statuses.append(result['status'])
            if result['status'] == 'failed':
                raise RuntimeError(result['error'])
            if result['status'] == 'done':
                DoneJob.model_validate(result)  # Includes JPEG decoding validation.
                elapsed = time.monotonic()-started
                return dict(video=video.name, success=elapsed < 300, elapsed_seconds=elapsed,
                            verdict=result['verdict'], events=len(result['violations']),
                            evidence_frames=sum(len(e['evidence_frames']) for e in result['violations']),
                            statuses=sorted(set(statuses)))
            if result['status'] != 'processing':
                raise ValueError('Invalid job status')
            time.sleep(min(.5, max(0, remaining)))
        raise TimeoutError('No successful result within 300 seconds')
    except Exception as exc:
        return dict(video=video.name, success=False, elapsed_seconds=time.monotonic()-started, error=str(exc))
    finally:
        session.close()


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--video', type=Path, nargs='+', required=True)
    parser.add_argument('--parallel', type=int, default=1, choices=range(1, 9))
    parser.add_argument('--allow-remote-upload', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('artifacts/jury_check.json'))
    args = parser.parse_args()
    parsed = urlparse(args.url)
    local = parsed.hostname in ('localhost', '127.0.0.1', '::1')
    if parsed.scheme not in ('http', 'https') or (not local and (not args.allow_remote_upload or parsed.scheme != 'https')):
        parser.error('Remote uploads require your own HTTPS service and --allow-remote-upload')
    key = os.getenv('FORTEVISION_API_KEY')
    if not key:
        parser.error('Set FORTEVISION_API_KEY; never put the secret in command arguments')
    session = requests.Session()
    session.trust_env = False
    url = args.url.rstrip('/')
    try:
        assert session.post(url+'/api/v1/analyze', timeout=10, allow_redirects=False).status_code == 401
        session.get(url+'/ready', timeout=15, allow_redirects=False).raise_for_status()
    finally:
        session.close()
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        reports = list(pool.map(lambda video: check_video(url, key, video), args.video))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'parallel': args.parallel, 'reports': reports}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(reports, ensure_ascii=True, indent=2))
    if not all(r['success'] for r in reports):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
