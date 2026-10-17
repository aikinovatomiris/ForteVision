"""Remove abandoned generated uploads while the API is stopped; dry-run by default."""
import argparse
import re
import sqlite3

from app.config import Settings
from app.runtime_lock import RuntimeLock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--purge-results', action='store_true')
    args = parser.parse_args()
    settings = Settings()
    lease = RuntimeLock(settings.job_db.with_suffix('.lock'))
    try:
        root = settings.upload_dir.resolve()
        candidates = [p for p in root.glob('*') if p.is_file() and not p.is_symlink()
                      and p.resolve().parent == root
                      and re.fullmatch(r'[0-9a-f]{32}\.(mov|result\.json(?:\.tmp)?)', p.name)]
        print(f'Generated files: {len(candidates)}; apply={args.apply}')
        if args.apply:
            for path in candidates:
                path.unlink()
        if args.purge_results and settings.job_db.exists():
            print('All job results selected for removal')
            if args.apply:
                with sqlite3.connect(settings.job_db) as db:
                    db.execute('PRAGMA secure_delete=ON')
                    db.execute('DELETE FROM jobs')
                    db.commit()
                    db.execute('VACUUM')
    finally:
        lease.close()


if __name__ == '__main__':
    main()
