"""Read-only diagnosis of the active artifact. No fitting or test-set tuning."""
import argparse
import json
import math
from pathlib import Path

from app.config import Settings, ROOT
from ml.dataset import write_json


def threshold_frontier(videos):
    """Historical validation scores only; this is not a fresh evaluation."""
    if not videos or {v['label'] for v in videos} != {0, 1}:
        return []
    thresholds = sorted({t for v in videos for t in
                         (float(v['score']), math.nextafter(float(v['score']), math.inf)) if 0 < t < 1})
    result = []
    for threshold in thresholds:
        tp = sum(v['label'] == 1 and v['score'] >= threshold for v in videos)
        fp = sum(v['label'] == 0 and v['score'] >= threshold for v in videos)
        fn = sum(v['label'] == 1 and v['score'] < threshold for v in videos)
        tn = sum(v['label'] == 0 and v['score'] < threshold for v in videos)
        result.append(dict(threshold=threshold, tp=tp, fp=fp, fn=fn, tn=tn,
                           precision=tp/(tp+fp) if tp+fp else None, recall=tp/(tp+fn)))
    return result


def audit(settings):
    meta_path = settings.model.with_suffix('.json')
    metadata = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
    required = ['models/action_states_v8.joblib', 'models/action_states_v8.json',
                'data/raw/team_annotations.json', 'data/annotations/actions_objects_v2.json']
    videos = metadata.get('validation_metrics', {}).get('videos', [])
    frontier = threshold_frontier(videos)
    full_recall = [p for p in frontier if p['fn'] == 0]
    return dict(active_model=settings.model.name,
        missing_release_files=[p for p in required if not (ROOT/p).exists()],
        image_mode=metadata.get('image_mode', 'center'), threshold=metadata.get('threshold'),
        historical_validation=metadata.get('validation_metrics', {}).get('video_metrics'),
        threshold_frontier=frontier,
        minimum_false_positives_at_full_validation_recall=min((p['fp'] for p in full_recall), default=None),
        limitations=['Historical validation was used for threshold selection; not an independent test.',
                    'Binary labels cannot supervise action names or descriptions.',
                    'Original Excel and release feedback require reconciliation before retraining.',
                    'No thresholds, models, labels or splits were modified.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/active_model_audit.json')
    args = parser.parse_args()
    report = audit(Settings())
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
