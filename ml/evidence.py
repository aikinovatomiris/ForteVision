import base64
from pathlib import Path

import cv2
import numpy as np

from .dataset import ForteError
from .preprocessing import check_deadline, read_at_times


def evidence_frames(path: Path, event, count=3, deadline=None):
    if not 1 <= count <= 3:
        raise ForteError('Количество evidence frames должно быть от 1 до 3')
    targets = [event['peak']]
    if count > 1:
        targets.append(max(event['start'], event['peak']-0.5))
    if count > 2:
        targets.append(min(event['end']-0.001, event['peak']+0.5))
    result = []
    for actual, frame in read_at_times(path, sorted(set(targets)), deadline):
        check_deadline(deadline)
        if not event['start'] <= actual < event['end']:
            continue
        if any(abs(r['timestamp_sec']-actual) < 1e-6 for r in result):
            continue
        scale = min(1.0, 960/max(frame.shape[:2]))
        if scale < 1:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise ForteError('JPEG encoder failed')
        result.append(dict(timestamp_sec=float(actual), image_base64=base64.b64encode(encoded.tobytes()).decode('ascii')))
    if not result:
        raise ForteError('Нет evidence frames внутри интервала события')
    return result
