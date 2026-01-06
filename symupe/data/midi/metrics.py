import math

import numpy as np
from symusic import Score


def pitch_class_entropy(midi: Score):
    pitches = []
    for track in midi.tracks:
        pitches.append(track.notes.numpy()["pitch"] % 12)
    pitches = np.concatenate(pitches)
    counts = np.array([np.sum(pitches == i) for i in range(12)])
    probs = counts / counts.sum()
    return -np.nansum(probs * np.log2(probs + 1e-5))


def _get_scale(root: int, mode: str) -> np.ndarray:
    """Return the scale mask for a specific root."""
    if mode == "major":
        c_scale = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], bool)
    elif mode == "minor":
        c_scale = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0], bool)
    else:
        raise ValueError("`mode` must be either 'major' or 'minor'.")
    return np.roll(c_scale, root)


def pitch_in_scale_rate(midi: Score, root: int, mode: str):
    scale = _get_scale(root, mode.lower())

    pitches = []
    for track in midi.tracks:
        pitches.append(track.notes.numpy()["pitch"] % 12)
    pitches = np.concatenate(pitches)

    note_count = len(pitches)
    in_scale_count = scale[pitches].sum()
    if note_count < 1:
        return math.nan
    return in_scale_count / note_count


def scale_consistency(midi: Score):
    max_in_scale_rate = 0.0
    for mode in ("major", "minor"):
        for root in range(12):
            rate = pitch_in_scale_rate(midi, root, mode)
            if math.isnan(rate):
                return math.nan
            if rate > max_in_scale_rate:
                max_in_scale_rate = rate
    return max_in_scale_rate
