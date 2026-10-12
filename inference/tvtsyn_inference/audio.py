"""Audio arrays use [samples] or [samples, channels] layout."""
import numpy as np


def mono(audio):
    x = np.asarray(audio)
    if x.ndim not in (1, 2) or (x.ndim == 2 and not 1 <= x.shape[1] <= 8):
        raise ValueError('Audio must have shape [samples] or [samples, channels] (1–8 channels)')
    if x.dtype.kind in 'iu':
        if x.dtype.kind == 'u':
            midpoint = float(2 ** (x.dtype.itemsize * 8 - 1))
            x = (x.astype(np.float32) - midpoint) / midpoint
        else:
            x = x.astype(np.float32) / float(2 ** (x.dtype.itemsize * 8 - 1))
    else:
        x = x.astype(np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if not np.isfinite(x).all():
        raise ValueError('Audio contains NaN or infinity')
    return np.ascontiguousarray(x)


def resample(audio, sample_rate, target_rate=16000):
    x = mono(audio)
    if not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
        raise ValueError('sample_rate must be a positive integer')
    if sample_rate != target_rate and x.size:
        import soxr
        x = soxr.resample(x, sample_rate, target_rate, quality='HQ')
    return x


def load_audio(path):
    import soundfile as sf
    return sf.read(path, dtype='float32', always_2d=False)


def speaker_array(value, width=704):
    if isinstance(value, (str, __import__('pathlib').Path)):
        value = np.load(value, allow_pickle=False)
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    x = np.asarray(value, dtype=np.float32)
    if x.shape == (width,):
        x = x[None, :]
    if x.shape != (1, width) or not np.isfinite(x).all():
        raise ValueError(f'Speaker embedding must be finite and have shape (1, {width})')
    return np.ascontiguousarray(x).copy()
