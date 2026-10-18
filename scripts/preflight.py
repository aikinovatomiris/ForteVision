"""Read-only local readiness check; never downloads, trains or prints secrets."""
import json
import sys
from importlib.metadata import version

def main():
    from app.config import Settings
    from app.readiness import model_ready
    from ml.dataset import sha256
    settings=Settings()
    report={'python':sys.version.split()[0], 'backend':settings.backend,
            'runtime_ready':model_ready(settings), 'automatic_detection_ready':False,
            'note':'Runtime readiness is not ML accuracy. Read review_candidates and assessment_status.'}
    report['packages']={name:version(name) for name in ('torch','torchvision','scikit-learn','numpy','fastapi','streamlit')}
    report['files']={p.name:dict(present=p.is_file(),sha256=sha256(p) if p.is_file() else None)
                     for p in (settings.model,settings.model.with_suffix('.json'),settings.weights)}
    report['key_configured']=bool(settings.api_key and not settings.api_key.startswith('replace-with-'))
    print(json.dumps(report,ensure_ascii=True,indent=2))
    return 0 if report['runtime_ready'] else 1

if __name__=='__main__':raise SystemExit(main())
