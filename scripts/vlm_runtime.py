"""Start the bundled, local-only vision server. Never downloads weights."""
import subprocess
import time
from pathlib import Path
from ml.local_vlm import ready
ROOT = Path(__file__).resolve().parents[1]

def start():
    if ready():
        return None
    executable = ROOT/'runtime/llama-b11716/llama-server.exe'
    model = ROOT/'models/vlm/Qwen3-VL-2B-Instruct-Q8_0.gguf'
    projector = ROOT/'models/vlm/mmproj-Qwen3-VL-2B-Instruct-Q8_0.gguf'
    for path in (executable, model, projector):
        if not path.is_file():
            raise RuntimeError(f'Local model file missing: {path}')
    log_path = ROOT/'artifacts/vlm_server.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('ab') as log:
        process = subprocess.Popen([str(executable), '-m', str(model), '--mmproj', str(projector),
            '--host', '127.0.0.1', '--port', '8091', '--ctx-size', '8192', '--parallel', '1',
            '--n-gpu-layers', '99', '--image-min-tokens', '512', '--image-max-tokens', '512',
            '--offline', '--no-webui', '--cors-origins', 'http://127.0.0.1:8501'],
            cwd=ROOT, stdout=log, stderr=log, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    try:
        deadline = time.monotonic()+180
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f'Vision server exited: see {log_path}')
            if ready():
                return process
            time.sleep(1)
        raise RuntimeError(f'Vision startup timeout: see {log_path}')
    except BaseException:
        process.terminate()
        process.wait(timeout=10)
        raise
