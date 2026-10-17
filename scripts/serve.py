"""Start API and UI together. No training, downloads or model changes."""
import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def runtime_environment(offline=True):
    env = os.environ.copy()
    env['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    if offline:
        env['PYTHONPATH'] = str(ROOT/'scripts/offline_guard')+os.pathsep+str(ROOT)+os.pathsep+env.get('PYTHONPATH', '')
        env['FORTEVISION_OFFLINE'] = '1'
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--api-port', type=int, default=8000)
    parser.add_argument('--ui-port', type=int, default=8501)
    args = parser.parse_args()
    if args.api_port == args.ui_port:
        parser.error('API и UI должны использовать разные порты')
    for port in (args.api_port, args.ui_port):
        try:
            with socket.socket() as probe:
                probe.bind((args.host, port))
        except OSError:
            parser.exit(2, f'Port {port} is busy; stop the old service or choose other ports.\n')
    env = runtime_environment()
    env['FORTEVISION_API_URL'] = f'http://127.0.0.1:{args.api_port}'
    commands = [
        [sys.executable, '-m', 'uvicorn', 'app.api:app', '--host', args.host, '--port', str(args.api_port), '--workers', '1', '--no-access-log'],
        [sys.executable, '-m', 'streamlit', 'run', 'app/frontend.py', '--server.address', args.host,
         '--server.port', str(args.ui_port), '--server.headless', 'true', '--browser.gatherUsageStats', 'false'],
    ]
    children = []
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    try:
        for command in commands:
            children.append(subprocess.Popen(command, cwd=ROOT, env=env))
        print(f'ForteVision UI: http://127.0.0.1:{args.ui_port} | API: http://127.0.0.1:{args.api_port}', flush=True)
        while all(child.poll() is None for child in children):
            time.sleep(.5)
        raise RuntimeError('One service stopped; shutting down its peer')
    except KeyboardInterrupt:
        pass
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=305)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == '__main__':
    main()
