import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT/'.env')


@dataclass
class Settings:
    api_key: str = field(default_factory=lambda: os.getenv('FORTEVISION_API_KEY', ''))
    model: Path = field(default_factory=lambda: Path(os.getenv('FORTEVISION_MODEL', str(ROOT/'models/classifier.joblib'))).resolve())
    weights: Path = field(default_factory=lambda: Path(os.getenv('FORTEVISION_WEIGHTS', str(ROOT/'models/resnet18-f37072fd.pth'))).resolve())
    device: str = field(default_factory=lambda: os.getenv('FORTEVISION_DEVICE', 'cpu'))
    upload_dir: Path = field(default_factory=lambda: Path(os.getenv('FORTEVISION_UPLOAD_DIR', str(Path(tempfile.gettempdir())/'fortevision-uploads'))).resolve())
    max_upload_mb: int = field(default_factory=lambda: int(os.getenv('FORTEVISION_MAX_UPLOAD_MB', '256')))
    queue_size: int = field(default_factory=lambda: int(os.getenv('FORTEVISION_QUEUE_SIZE', '3')))
    max_jobs: int = field(default_factory=lambda: int(os.getenv('FORTEVISION_MAX_JOBS', '20')))
    ttl_seconds: int = field(default_factory=lambda: int(os.getenv('FORTEVISION_JOB_TTL_SECONDS', '3600')))
    timeout_seconds: int = field(default_factory=lambda: int(os.getenv('FORTEVISION_TIMEOUT_SECONDS', '295')))
    job_db: Path | None = field(default_factory=lambda: Path(os.environ['FORTEVISION_JOB_DB']).resolve() if os.getenv('FORTEVISION_JOB_DB') else None)
    rules: Path = field(default_factory=lambda: Path(os.getenv('FORTEVISION_RULES', str(ROOT/'configs/rules.json'))).resolve())

    def __post_init__(self):
        if self.device not in ('cpu', 'cuda'):
            raise ValueError('FORTEVISION_DEVICE должен быть cpu или cuda')
        if self.job_db is None:
            self.job_db = self.upload_dir.parent / (self.upload_dir.name+'-state') / 'jobs.sqlite3'
        if min(self.max_upload_mb, self.queue_size, self.max_jobs, self.ttl_seconds, self.timeout_seconds) <= 0:
            raise ValueError('Лимиты должны быть положительными')
        if self.timeout_seconds > 295:
            raise ValueError('Timeout должен быть <= 295 секунд')

    @property
    def max_upload_bytes(self):
        return self.max_upload_mb*1024*1024
