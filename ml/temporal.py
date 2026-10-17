import numpy as np


def merge_events(windows, probabilities, threshold):
    """Merge touching positive windows; peak uses a real sampled PTS."""
    events = []
    for window, score in zip(windows, probabilities):
        if score < threshold:
            continue
        if events and window['start'] <= events[-1]['end'] + 1e-6:
            current = events[-1]
            current['end'] = max(current['end'], window['end'])
            if score > current['score']:
                current.update(score=float(score), peak=window['timestamp'])
        else:
            events.append(dict(start=window['start'], end=window['end'], peak=window['timestamp'], score=float(score)))
    return events


def window_labels(windows, label, timestamp, positive_radius=1.0, exclusion_zone=6.0, weak_negatives=False):
    """-1 = unknown. Weak labels are separate from observed negatives."""
    labels, weak = [], []
    events = timestamp if isinstance(timestamp, (list, tuple)) else ([] if timestamp is None else [timestamp])
    for w in windows:
        positive = label == 1 and any(w['start'] <= t + positive_radius and w['end'] > t - positive_radius for t in events)
        distant = label == 1 and bool(events) and all(w['end'] <= t - exclusion_zone or w['start'] >= t + exclusion_zone for t in events)
        labels.append(0 if label == 0 else 1 if positive else 0 if weak_negatives and distant else -1)
        weak.append(bool(weak_negatives and distant and not positive))
    return np.asarray(labels), np.asarray(weak)
