import math

import numpy as np
from symusic import Score


def pitch_class_entropy(midi: Score):
    """Computes Shannon entropy of pitch class distribution.

    Args:
        midi: :class:`symusic.Score` object.

    Returns:
        Entropy value in bits; higher values indicate higher chromatic complexity.
    """
    pitches = []
    for track in midi.tracks:
        pitches.append(track.notes.numpy()["pitch"] % 12)
    pitches = np.concatenate(pitches)
    counts = np.array([np.sum(pitches == i) for i in range(12)])
    probs = counts / counts.sum()
    return -np.nansum(probs * np.log2(probs + 1e-5))


def _get_scale(root: int, mode: str) -> np.ndarray:
    """Generates boolean mask for specific musical scale.

    Args:
        root: Pitch class index of scale root (0-11).
        mode: Scale type, either 'major' or 'minor'.

    Returns:
        Boolean array of size 12 representing pitch classes belonging to scale.
    """
    if mode == "major":
        c_scale = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], bool)
    elif mode == "minor":
        c_scale = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0], bool)
    else:
        raise ValueError("`mode` must be either 'major' or 'minor'.")
    return np.roll(c_scale, root)


def pitch_in_scale_rate(midi: Score, root: int, mode: str):
    """Calculates ratio of notes belonging to a specific musical scale.

    Args:
        midi: :class:`symusic.Score` object.
        root: Root of scale (0-11).
        mode: Scale type ('major' or 'minor').

    Returns:
        Float ratio (0.0 to 1.0).
    """
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
    """Finds best-fit scale for MIDI and returns corresponding in-scale rate.

    Checks all 12 roots for both major and minor modes.

    Args:
        midi: :class:`symusic.Score` object.

    Returns:
        Maximum in-scale rate found across all scales.
    """
    max_in_scale_rate = 0.0
    for mode in ("major", "minor"):
        for root in range(12):
            rate = pitch_in_scale_rate(midi, root, mode)
            if math.isnan(rate):
                return math.nan
            if rate > max_in_scale_rate:
                max_in_scale_rate = rate
    return max_in_scale_rate
