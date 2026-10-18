"""Subprocess entry point: isolates decoder hangs from the HTTP worker."""
import argparse
import json
from pathlib import Path

from .dataset import write_json



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['classifier', 'local_vlm'], default='classifier')
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=Path('models/classifier.joblib'))
    parser.add_argument('--weights', type=Path, default=Path('models/resnet18-f37072fd.pth'))
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--budget', type=float, default=285)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        if args.backend == 'local_vlm':
            from .local_vlm import analyze
            result = analyze(args.video, args.budget)
        else:
            from .pipeline import analyze
            result = analyze(args.video, args.model, args.weights, args.device, args.budget)
        if args.output:
            write_json(args.output, result)
        else:
            print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    except Exception as exc:
        parser.exit(2, json.dumps({'error': str(exc)}, ensure_ascii=True)+'\n')


if __name__ == '__main__':
    main()
