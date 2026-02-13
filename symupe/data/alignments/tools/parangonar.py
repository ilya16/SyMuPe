from __future__ import annotations

import os
import resource
import signal
import time
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import numba as nb
import parangonar as pa
import parangonar.match.matchers
import partitura as pt
from parangonar.dp.dtw import dtw_backtracking
from symusic import Score

from symupe.utils import ExplicitEnum
from .aligner import Aligner
from ..alignment import Alignment, AlignmentNote, AlignmentPair
from ...midi.preprocess import preprocess_midi
from ...partitura.utils import load_performance_note_array


class DataType(ExplicitEnum):
    SCORE = "score"
    PERFORMANCE = "perf"
    ALIGNMENT = "align"


class FileType(ExplicitEnum):
    MIDI = "MIDI"
    ALIGN = "align"


FILE_EXT = {
    FileType.MIDI: ".mid",
    FileType.ALIGN: "_align.txt"
}


@contextmanager
def time_limit(seconds):
    if seconds < 0:
        raise TimeoutError

    def signal_handler(signum, frame):
        raise TimeoutError

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        signal.alarm(0)


class ParangonarAligner(Aligner):
    """ Wrapper for Parangonar Alignment Tool.

    Links:
        https://github.com/sildater/parangonar
    """

    def __init__(
            self,
            midi_dir: str | Path | None = None,
            alignments_dir: str | Path | None = None
    ):
        self.midi_dir = Path(midi_dir) if alignments_dir is not None else None
        self.alignments_dir = Path(alignments_dir) if alignments_dir is not None else None

        self.matcher = pa.DualDTWNoteMatcher()
        self.matcher.onset_matcher.dtw = DynamicTimeWarpingSingleLoop()

    def align(
            self,
            score_midi: str | Path,
            perf_midi: str | Path,
            score_note_array: np.ndarray | None = None,
            perf_note_array: np.ndarray | None = None,
            process_ornaments: bool = True,
            save_alignment: bool = False,
            timeout: float | None = 1000.,
            memory_limit: int | None = None
    ):
        start_time = time.perf_counter()

        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        if memory_limit is not None:
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit, hard))

        paths = {DataType.ALIGNMENT.value: {}}

        time_left = int(timeout - (time.perf_counter() - start_time))
        try:
            with warnings.catch_warnings(), time_limit(time_left):
                warnings.simplefilter("ignore")
                score = pt.load_score_midi(filename=score_midi, part_voice_assign_mode=4, estimate_voice_info=False)

                if score_note_array is None:
                    score_note_array = score.note_array(include_grace_notes=True)
            score_note_array["id"] = [f"n{i}" for i in range(len(score_note_array))]
        except TimeoutError:
            msg = f"TimeoutExpired@pt.load_score_midi: computation of the score note array " \
                  f"exceeded time limit of {time_left:.2f}s"
            return None, paths, [msg]
        except Exception as e:
            msg = f"Error@pt.load_score_midi: computation of the score note array failed with ({repr(e)})"
            return None, paths, [msg]

        time_left = int(timeout - (time.perf_counter() - start_time))
        try:
            with warnings.catch_warnings(), time_limit(time_left):
                warnings.simplefilter("ignore")
                if perf_note_array is None:
                    perf_note_array = load_performance_note_array(midi_path=perf_midi)
        except TimeoutError:
            msg = f"TimeoutExpired@pt.load_performance_midi: computation of the performance note array " \
                  f"exceeded time limit of {time_left:.2f}s"
            return None, paths, [msg]
        except Exception as e:
            msg = f"Error@pt.load_performance_midi: computation of the performance note array failed with ({repr(e)})"
            return None, paths, [msg]

        time_left = int(timeout - (time.perf_counter() - start_time))
        try:
            with time_limit(time_left):
                if len(score_note_array) == len(perf_note_array) == 1:  # parangonar hangs on one note pairs
                    if score_note_array[0]["pitch"] == perf_note_array[0]["pitch"]:
                        pred_alignment = [{"label": "match", "score_id": "n0", "performance_id": "n0"}]
                    else:
                        pred_alignment = [
                            {"label": "deletion", "score_id": "n0"},
                            {"label": "insertion", "performance_id": "n0"},
                        ]
                else:
                    pred_alignment = self.compute_alignment(
                        score_note_array,
                        perf_note_array,
                        score,
                        process_ornaments=process_ornaments
                    )
        except TimeoutError:
            msg = f"TimeoutExpired@compute_alignment: computation of the alignment " \
                  f"exceeded time limit of {time_left:.2f}s"
            return None, paths, [msg]
        except Exception as e:
            msg = f"Error@compute_alignment: computation of the alignment failed with ({repr(e)})"
            return None, paths, [msg]

        # prepare alignment name and path
        if self.midi_dir is not None:
            perf_name = os.path.relpath(perf_midi, self.midi_dir).replace(FILE_EXT[FileType.MIDI], "")
            score_name = os.path.relpath(score_midi, self.midi_dir).replace(FILE_EXT[FileType.MIDI], "")
        else:
            perf_name, score_name = os.path.basename(perf_midi), os.path.basename(score_midi)
        score_short_name = os.path.basename(score_name)

        align_name = f"{perf_name}_{score_short_name}"

        # process and save alignment
        try:
            score_midi = preprocess_midi(
                Score(score_midi),
                cut_overlapped_notes=True,
                clean_duplicates=True
            ).to("second")

            pairs = []
            for x in pred_alignment:
                score_note = None
                if "score_id" in x:
                    idx = int(x["score_id"][1:])
                    note = score_note_array[idx]
                    score_note = AlignmentNote(
                        idx=idx,
                        pitch=int(note["pitch"]),
                        start=round(float(score_midi.tracks[0].notes[idx].start), 6),
                        end=round(float(score_midi.tracks[0].notes[idx].end), 6)
                    )

                perf_note = None
                if "performance_id" in x:
                    idx = int(x["performance_id"][1:])
                    note = perf_note_array[idx]
                    perf_note = AlignmentNote(
                        idx=idx,
                        pitch=int(note["pitch"]),
                        start=round(float(note["onset_sec"]), 6),
                        end=round(float(note["onset_sec"]) + float(note["duration_sec"]), 6)
                    )

                pairs.append(AlignmentPair(score_note=score_note, perf_note=perf_note))

            alignment = Alignment(path=None, pairs=pairs, score_name=score_name, perf_name=perf_name)

            if save_alignment:
                assert self.alignments_dir is not None
                align_path = self.alignments_dir / (align_name + FILE_EXT[FileType.ALIGN])
                paths[DataType.ALIGNMENT][FileType.ALIGN.value] = align_path

                os.makedirs(os.path.dirname(align_path), exist_ok=True)

                alignment.preprocess_pairs(sort=True, score_first=False, clean_duplicates=True)
                alignment.write(str(align_path))
                alignment.preprocess_pairs(sort=True, score_first=True, clean_duplicates=False)

        except Exception as e:
            msg = f"Error@process_alignment: post-processing of the alignment failed with ({repr(e)})"
            return None, paths, [msg]

        return alignment, paths, []

    def compute_alignment(
            self,
            score_note_array,
            perf_note_array,
            score: pt.score.Score | None = None,
            process_ornaments: bool = True
    ):
        return self.matcher(
            score_note_array,
            perf_note_array,
            process_ornaments=process_ornaments and score is not None,
            score_part=score[0] if score is not None else None
        )  # if a score part is passed, ornaments can be handled separately


class DynamicTimeWarping(object):
    """
    pure python vanilla Dynamic Time Warping
    """

    def __init__(self, metric="euclidean"):
        self.metric = metric

    def __call__(self, X, Y, return_path=True, return_cost_matrix=False):
        X = np.asanyarray(X, dtype=float)
        Y = np.asanyarray(Y, dtype=float)
        # Compute pairwise distance
        D = cdist(X, Y)
        # Compute accumulated cost matrix
        dtwd_matrix = dtw_dmatrix_from_pairwise_dmatrix(D)
        dtwd_distance = dtwd_matrix[-1, -1]

        # Output
        out = (dtwd_distance,)

        if return_path:
            # Compute alignment path
            path = dtw_backtracking(dtwd_matrix)
            out += (path,)
        if return_cost_matrix:
            out += (dtwd_matrix,)
        return out


class DynamicTimeWarpingSingleLoop(object):
    """
    pure python vanilla Dynamic Time Warping
    """
    def __call__(
            self,
            X, Y,
            return_path=True,
            return_cost_matrix=False
    ):
        # Compute the pw distances and accumulated cost matrix
        Y1 = np.full((len(Y), max(map(len, Y))), fill_value=-1)
        for i, y in enumerate(Y):
            Y1[i, :len(y)] = list(y)

        dtwd_matrix = cdist_dtw_single_loop(X, Y1)

        # dtwd_matrix = dtw_dmatrix_from_pairwise_dmatrix(D)
        dtwd_distance = dtwd_matrix[-1, -1]

        # Output
        out = (dtwd_distance,)

        if return_path:
            # Compute alignment path
            path = dtw_backtracking(dtwd_matrix)
            out += (path,)
        if return_cost_matrix:
            out += (dtwd_matrix,)
        return out


@nb.njit
def isin(a, b):
    out = np.empty(a.shape[0], dtype=nb.boolean)
    b = set(b)
    for i in nb.prange(a.shape[0]):
        if a[i] in b:
            out[i] = True
        else:
            out[i] = False
    return out


@nb.njit
def cdist_dtw_single_loop(arr1, arr2):
    """

    compute  a pairwise distance matrix
    and its dynamic time warping cost matrix

    Parameters
    ----------

    arr1: numpy nd array or list

    arr2: numpy nd array or list

    metric> callable
        a metric function

    Returns
    -------
    dtwd : np.ndarray
        Accumulated cost matrix
    """
    # if arrays and helper variables
    M = len(arr1)  # arr1.shape[0]
    N = len(arr2)  # arr2.shape[0]

    dtwd = np.ones((M + 1, N + 1), dtype=float) * np.inf

    for j in range(1, N + 1):
        dtwd[1:, j] = ~isin(arr1, arr2[j - 1])

    # Compute the distance iteratively
    dtwd[0, 0] = 0
    for i in range(1, M + 1):
        for j in range(1, N + 1):
            insertion = dtwd[i - 1, j]
            deletion = dtwd[i, j - 1]
            match = dtwd[i - 1, j - 1]
            dtwd[i, j] = dtwd[i, j] + min((insertion, deletion, match))

    return dtwd[1:, 1:]  # pdist_array


@nb.njit
def dtw_dmatrix_from_pairwise_dmatrix(D):
    """
    compute dynamic time warping cost matrix
    from a pairwise distance matrix

    Parameters
    ----------
    D : double array
        Pairwise distance matrix (computed e.g., with `cdist`).

    Returns
    -------
    dtwd : np.ndarray
        Accumulated cost matrix
    """
    # Initialize arrays and helper variables
    M = D.shape[0]
    N = D.shape[1]
    # the dtwd distance matrix is initialized with INFINITY
    dtwd = np.ones((M + 1, N + 1), dtype=float) * np.inf

    # Compute the distance iteratively
    dtwd[0, 0] = 0
    for i in range(1, M + 1):
        for j in range(1, N + 1):
            c = D[i - 1, j - 1]
            insertion = dtwd[i - 1, j]
            match = dtwd[i - 1, j - 1]
            deletion = dtwd[i, j - 1]
            dtwd[i, j] = c + min((insertion, deletion, match))

    return dtwd[1:, 1:]


def cdist(XA, XB):
    """
    Compute the Euclidean distance between each pair of points in XA and XB using NumPy vectorization.

    Parameters
    ----------
    XA : np.ndarray of shape (m, d)
        First set of m points in d-dimensional space.
    XB : np.ndarray of shape (n, d)
        Second set of n points in d-dimensional space.

    Returns
    -------
    D : np.ndarray of shape (m, n)
        Distance matrix where D[i, j] is the Euclidean distance between XA[i] and XB[j].
    """
    return np.sqrt(((XA[:, np.newaxis, :] - XB[np.newaxis, :, :]) ** 2).sum(axis=2))


def unique_alignments(xs, ys, threshold=None):
    """
    From two sequences of numbers, return the unique ID
    tuples of aligned values that minimize the sum of
    tupel distances.

    Parameters
    ----------
    xs : np.array
        Sequence of numbers
    ys : np.array
        Sequence of numbers
    threshold : float

    Returns
    -------
    tuples : list

    """
    matcher = DynamicTimeWarping()
    _, p = matcher(xs.reshape((-1, 1)), ys.reshape((-1, 1)), return_path=True)

    used_x = set()
    tuples = list()
    for x in xs:
        if not x in used_x:
            current_x_mask = xs[p[:, 0]] == x
            if current_x_mask.sum() > 1:
                current_ids = p[current_x_mask, :]
                current_y = ys[current_ids[:, 1]]
                current_x = xs[current_ids[:, 0]]

            else:
                y = ys[p[current_x_mask, 1]]
                current_y_mask = ys[p[:, 1]] == y
                if current_y_mask.sum() > 1:
                    current_ids = p[current_y_mask, :]
                    current_y = ys[current_ids[:, 1]]
                    current_x = xs[current_ids[:, 0]]

                else:
                    current_ids = p[current_x_mask, :].reshape(1, 2)
                    current_x = x
                    current_y = y

            candidate_dist = np.min(np.abs(current_x - current_y))
            candidate_id = np.argmin(np.abs(current_x - current_y))
            if threshold is None or candidate_dist < threshold:
                tuples.append(
                    (current_ids[candidate_id, 0], current_ids[candidate_id, 1])
                )
            used_x.update(set(np.unique(current_x)))
    return tuples

parangonar.match.matchers.unique_alignments = unique_alignments
