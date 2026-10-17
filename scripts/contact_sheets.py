"""Local contact sheets around annotations, for manual examination only."""
import argparse
from pathlib import Path

import cv2
import numpy as np

from ml.dataset import ForteError, key, read_labels, write_json
from ml.preprocessing import probe, read_at_times


def create_sheet(path, timestamp, output, radius=2.0):
    info = probe(path)
    if not 0 <= timestamp < info['duration_seconds']:
        raise ForteError('Timestamp вне видео')
    requested = np.linspace(max(0, timestamp-radius), min(info['duration_seconds']-1/info['fps'], timestamp+radius), 9)
    cells, actual_times = [], []
    for actual, frame in read_at_times(path, requested):
        canvas = np.zeros((220, 320, 3), dtype=np.uint8)
        ratio = min(320/frame.shape[1], 190/frame.shape[0])
        thumbnail = cv2.resize(frame, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
        canvas[:thumbnail.shape[0], :thumbnail.shape[1]] = thumbnail
        cv2.putText(canvas, f'actual {actual:.3f} s', (8, 212), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cells.append(canvas)
        actual_times.append(actual)
    sheet = np.vstack([np.hstack(cells[i:i+3]) for i in (0, 3, 6)])
    ok, encoded = cv2.imencode('.jpg', sheet)
    if not ok:
        raise ForteError('Не удалось создать JPEG contact sheet')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(encoded.tobytes())  # Unicode-safe on Windows
    write_json(output.with_suffix('.json'), dict(file=path.name, annotated_timestamp=timestamp, actual_timestamps=actual_times))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', type=Path, default=Path('data/raw/open_labels.xlsx'))
    parser.add_argument('--videos', type=Path, default=Path('data/videos'))
    parser.add_argument('--output', type=Path, default=Path('research/contact_sheets'))
    parser.add_argument('--video-id')
    parser.add_argument('--radius', type=float, default=2)
    args = parser.parse_args()
    if not np.isfinite(args.radius) or args.radius < 0:
        parser.error('radius должен быть конечным и >= 0')
    try:
        for row in read_labels(args.labels, args.videos):
            if row['timestamp'] is not None and (not args.video_id or row['id'] == key(args.video_id)):
                create_sheet(args.videos/row['file'], row['timestamp'], args.output/f"{row['id']}.jpg", args.radius)
                print(f"Saved {row['id']}")
    except (ForteError, OSError) as exc:
        parser.exit(2, ascii(str(exc))+'\n')


if __name__ == '__main__':
    main()
