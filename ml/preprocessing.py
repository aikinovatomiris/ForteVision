from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from .dataset import ForteError


@dataclass(frozen=True)
class SamplingConfig:
    fps: float = 2.0
    window_seconds: float = 4.0
    stride_seconds: float = 2.0
    batch_size: int = 16
    max_duration_seconds: float = 3600.0

    def __post_init__(self):
        numeric = (self.fps, self.window_seconds, self.stride_seconds, self.max_duration_seconds)
        if not all(np.isfinite(v) and v > 0 for v in numeric):
            raise ForteError('Параметры sampling должны быть положительными и конечными')
        if self.stride_seconds > self.window_seconds or self.fps > 30 or not 1 <= self.batch_size <= 256:
            raise ForteError('Нужны stride <= window, FPS <= 30, batch_size 1..256')

    def to_dict(self):
        return asdict(self)


def check_deadline(deadline):
    if deadline is not None and perf_counter() >= deadline:
        raise ForteError('Превышен бюджет обработки видео; предсказание не сформировано')


def probe(path: Path):
    if not path.is_file():
        raise ForteError(f'Видео не найдено: {path.name}')
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ForteError(f'Невозможно открыть видео/кодек: {path.name}')
        fps, count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if not np.isfinite([fps, count, width, height]).all() or min(fps, count, width, height) <= 0:
            raise ForteError('Некорректные FPS, длительность или разрешение')
        return dict(fps=float(fps), frame_count=int(count), duration_seconds=count/fps, width=int(width), height=int(height))
    finally:
        cap.release()


def iter_frames(path: Path, config: SamplingConfig, deadline=None):
    """Yield BGR frames and decoder POS_MSEC, rejecting broken/non-monotonic clocks."""
    info = probe(path)
    if info['duration_seconds'] > config.max_duration_seconds:
        raise ForteError('Видео превышает лимит длительности')
    cap = cv2.VideoCapture(str(path))
    previous, next_sample, count, selected = -float('inf'), 0.0, 0, 0
    try:
        while True:
            check_deadline(deadline)
            ok, frame = cap.read()
            if not ok:
                break
            count += 1
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if not np.isfinite(timestamp) or (count > 1 and timestamp <= previous):
                raise ForteError('Декодер не предоставляет возрастающие timestamps; нужен поддерживаемый MOV/кодек')
            previous = timestamp
            if timestamp < 0:
                continue  # MOV preroll; preserve the original clock, do not shift PTS.
            if timestamp + 1e-6 >= next_sample:
                selected += 1
                yield timestamp, frame
                next_sample = timestamp + 1.0 / config.fps
        if count < info['frame_count'] - 1:
            raise ForteError(f'Видео обрывается: декодировано {count} из {info["frame_count"]} кадров')
        if selected < 2:
            raise ForteError('Недостаточно кадров для временного анализа')
    finally:
        cap.release()


def pooled_windows(timestamps, embeddings, config: SamplingConfig, duration):
    times = np.asarray(timestamps, dtype=float)
    vectors = np.asarray(embeddings, dtype=np.float32)
    if len(times) < 2 or vectors.ndim != 2 or len(vectors) != len(times):
        raise ForteError('Недостаточно embeddings или неверная форма')
    if not np.isfinite(vectors).all():
        raise ForteError('Некорректные embeddings')
    windows, features = [], []
    for start in np.arange(0, duration, config.stride_seconds):
        end = min(float(start + config.window_seconds), duration)
        mask = (times >= start) & (times < end)
        if not mask.any():
            continue
        chunk, actual_times = vectors[mask], times[mask]
        windows.append(dict(start=float(start), end=float(end), timestamp=float(actual_times[len(actual_times)//2]), frame_timestamps=actual_times.tolist()))
        features.append(np.concatenate((chunk.mean(axis=0), chunk.max(axis=0))))
    if not windows:
        raise ForteError('Не удалось сформировать временные окна')
    return np.asarray(features), windows


def read_at_times(path: Path, targets, deadline=None):
    """Decode once and select nearest real PTS. Some MOVs decode but cannot seek."""
    targets = sorted(float(t) for t in targets)
    if not targets or any(not np.isfinite(t) or t < 0 for t in targets):
        raise ForteError('Некорректные целевые timestamps')
    info = probe(path)
    cap = cv2.VideoCapture(str(path))
    result, previous, cursor = [], None, 0
    try:
        while cursor < len(targets):
            check_deadline(deadline)
            ok, frame = cap.read()
            if not ok:
                if previous is not None and targets[-1] <= previous[0]+1/info['fps']:
                    result.extend([previous]*(len(targets)-cursor))
                    break
                raise ForteError('Видео закончилось до запрошенных evidence timestamps')
            actual = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
            if not np.isfinite(actual) or (previous is not None and actual <= previous[0]):
                raise ForteError('Некорректная временная шкала evidence')
            if actual < 0:
                continue  # Ignore preroll frames before video time zero.
            while cursor < len(targets) and actual >= targets[cursor]:
                chosen = (actual, frame)
                if previous is not None and abs(previous[0]-targets[cursor]) < abs(actual-targets[cursor]):
                    chosen = previous
                result.append(chosen)
                cursor += 1
            previous = (actual, frame)
        return result
    finally:
        cap.release()
