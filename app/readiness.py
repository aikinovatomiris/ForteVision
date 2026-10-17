"""Cached integrity checks, independent of prediction or training."""
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=4)
def _check(model, metadata, weights, signatures, device):
    from ml.pipeline import load_classifier
    from ml.dataset import sha256
    _, config = load_classifier(Path(model))
    if sha256(Path(weights)) != config['weights_sha256']:
        raise ValueError('Веса не соответствуют метаданным модели')
    if device == 'cuda':
        import torch
        if not torch.cuda.is_available():
            raise ValueError('CUDA недоступна')
    return True


def model_ready(settings):
    if not settings.api_key or settings.api_key.startswith('replace-with-'):
        return False
    paths = (settings.model, settings.model.with_suffix('.json'), settings.weights)
    try:
        signatures = tuple((p.stat().st_size, p.stat().st_mtime_ns) for p in paths)
        return _check(*(str(p) for p in paths), signatures, settings.device)
    except Exception:
        return False
