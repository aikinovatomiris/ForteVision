"""Real inference with Python egress disabled; never trains or downloads."""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from app.config import Settings
from ml.dataset import sha256
from scripts.serve import ROOT, runtime_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('artifacts/offline_verification.json'))
    args = parser.parse_args()
    settings = Settings()
    protected = [settings.model, settings.model.with_suffix('.json'), settings.weights]
    before = {str(p): sha256(p) for p in protected}
    env = runtime_environment()
    probe = """import os,socket
assert os.environ.get('FORTEVISION_OFFLINE_GUARD')=='active'
for action in (lambda: socket.create_connection(('203.0.113.1',443),timeout=.1), lambda: socket.getaddrinfo('example.invalid',443)):
    try: action()
    except PermissionError: pass
    else: raise AssertionError('Egress guard is inactive')
print('Python egress guard verified')
"""
    subprocess.run([sys.executable, '-c', probe], env=env, cwd=ROOT, check=True, timeout=15)
    with tempfile.TemporaryDirectory(prefix='fortevision-offline-') as directory:
        output = Path(directory)/'result.json'
        subprocess.run([sys.executable, '-m', 'ml.predict', '--video', str(args.video.resolve()),
                        '--model', str(settings.model), '--weights', str(settings.weights), '--device', settings.device,
                        '--output', str(output)], env=env, cwd=ROOT, check=True, timeout=295)
        result = json.loads(output.read_text(encoding='utf-8'))
        from app.schemas import DoneJob
        DoneJob(job_id='offline-check', status='done', **result)
    unchanged = before == {str(p): sha256(p) for p in protected}
    assert unchanged
    report = dict(python_egress_blocked=True, os_network_isolation_tested=False,
                  model_unchanged=unchanged, verdict=result['verdict'], elapsed_seconds=result['elapsed_seconds'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
