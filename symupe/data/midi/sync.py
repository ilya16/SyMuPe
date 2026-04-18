from dataclasses import dataclass

import numpy as np
from symusic import Score, Track, TimeSignature, Tempo, Note, TextMeta
from symusic.core import Tick

from symupe.utils import ExplicitEnum
from .beats import get_inter_beat_interval, get_bar_beat_ticks, get_performance_beats
from .timing import MIDITimeMapper
from .utils import filter_extra_midi_events, resample_midi


class GridLevel(ExplicitEnum):
    """Granularity level for timing synchronization."""

    BEAT = "beat"
    BAR = "bar"


@dataclass
class SyncMetadata:
    """Metadata capturing results of the synchronization process."""

    initial_shift: float
    grid_level: str
    num_points: int
    ticks_per_quarter: int
    time_signatures: list[TimeSignature]
    ticks_per_bar: np.ndarray | None = None
    ibi_in_quarters: np.ndarray | None = None
    tempos: np.ndarray | None = None


def sync_performance_midi(
    score_midi: Score,
    perf_midi: Score,
    onset_pairs: np.ndarray,
    synchronize_grid: bool = True,
    grid_level: str | GridLevel = GridLevel.BAR,
    ticks_per_quarter: int = 480,
    inplace: bool = True,
) -> tuple[Score, SyncMetadata | None]:
    """Refines performance MIDI timing by aligning it with the score's beat or bar structure.

    It operates in two primary modes:

    1. **Structural Synchronization** (`sync_to_score_grid=True`):
       Calculates a beat-to-time mapping using the provided onset pairs. It
       inserts inter-beat tempo changes into the performance MIDI so that its
       beat structure matches the symbolic score grid perfectly.

    2. **Timeline Realignment** (`sync_to_score_grid=False`):
       Simply shifts the performance timeline so the first aligned note matches
       the score's start, preserving the original human tempo and expressive
       timing without warping.

    Args:
        score_midi: The score providing the symbolic beat grid.
        perf_midi: The performance to be synchronized.
        onset_pairs: A 2D array of matched (score_tick, perf_time) pairs.
        synchronize_grid: If ``True``, warps performance timing to fit the score
            grid via tempo changes. If ``False``, performs a linear shift only.
        grid_level: The granularity of the synchronization grid.
            ``GridLevel.BAR`` (or "bar") for measure-level synchronization
            ``GridLevel.BEAT`` (or "beat") for beat-level synchronization.
        ticks_per_quarter: Target MIDI resolution (TPQ). The performance will be resampled to this value.
        inplace: If ``True``, modifies the `perf_midi` object in-place.

    Returns:
        A synchronized symusic.Score object with updated tempo/timing structure
        and copied time signatures from the score.
    """
    perf_midi = perf_midi if inplace else perf_midi.copy()
    ttype = "tick" if isinstance(perf_midi.ttype, Tick) else "second"
    time_mapper = MIDITimeMapper(perf_midi) if ttype == "tick" else None

    # prepare metadata
    ticks_per_bar = ibi_in_quarters = tempos = None

    # preprocess performance midi
    filter_extra_midi_events(perf_midi, use_sustain_boundaries=True)

    # process symbolic note markings if they are present, create tracks of marker notes
    marker_tracks = {}
    markers = []
    for marker in perf_midi.markers:
        if marker.text.startswith("Note") and marker.text[5] == "_":
            note_marker, pitch, start, duration = marker.text.split("_")
            pitch = int(pitch)
            start0, duration = map(float, (start, duration))
            start = time_mapper.s2t(start0) if time_mapper is not None else start0
            duration = (
                time_mapper.s2t(start0 + duration) - start if time_mapper is not None else duration
            )

            if note_marker not in marker_tracks:
                marker_tracks[note_marker] = Track(
                    program=0, name=f"Marker {note_marker}", ttype=ttype
                )
            marker_tracks[note_marker].notes.append(
                Note(time=start, duration=duration, pitch=pitch, velocity=0, ttype=ttype)
            )
        else:
            markers.append(marker)
    for track in marker_tracks.values():
        perf_midi.tracks.append(track)
    perf_midi.markers = markers

    # get absolute timing of tracks
    perf_midi_s = perf_midi.to("second")

    # compute score and performance onsets
    score_bars, score_beats = get_bar_beat_ticks(score_midi)
    score_onsets = score_bars if grid_level == GridLevel.BAR else score_beats
    score_onsets, perf_onsets = get_performance_beats(
        score_onsets, onset_pairs, monotonic_times=True, ticks_per_quarter=ticks_per_quarter
    )
    perf_shift = min(
        perf_onsets[0], min([track.notes.numpy()["time"].min() for track in perf_midi_s.tracks])
    )

    # shift onsets and performance events
    perf_onsets -= perf_shift
    perf_midi_s.shift_time(-perf_shift, inplace=True)

    perf_midi_s.markers.insert(
        0, TextMeta(time=0, text=f"ShiftSync_{perf_shift:.6f}", ttype="second")
    )

    if synchronize_grid:
        # align onsets to score grid
        time_sig_ticks, quarter_note_factors, inter_onset_intervals = [], [], []
        for time_sig in score_midi.time_signatures:
            time_sig_ticks.append(time_sig.time)
            quarter_note_factors.append(4 * time_sig.numerator / time_sig.denominator)
            inter_onset_intervals.append(
                get_inter_beat_interval(
                    time_sig=time_sig, ticks_per_quarter=score_midi.ticks_per_quarter
                )
            )

        time_sig_ticks, quarter_note_factors, inter_onset_intervals = map(
            np.array, (time_sig_ticks, quarter_note_factors, inter_onset_intervals)
        )
        inter_beat_intervals = inter_onset_intervals

        ticks_per_bar = (score_midi.ticks_per_quarter * quarter_note_factors).astype(int)
        beats_per_bar = ticks_per_bar / inter_beat_intervals
        ioi_in_quarters = ibi_in_quarters = quarter_note_factors / beats_per_bar

        if grid_level == GridLevel.BAR:
            inter_onset_intervals = inter_onset_intervals * beats_per_bar
            ioi_in_quarters = ioi_in_quarters * beats_per_bar

        # compute tempos
        intervals = np.diff(perf_onsets)
        if np.any(intervals <= 0.0):
            return perf_midi, None

        time_sig_indices = (np.searchsorted(time_sig_ticks, score_onsets, side="right") - 1)[:-1]
        inter_onset_ratios = np.diff(score_onsets) / inter_onset_intervals[time_sig_indices]
        tempos = 60 / intervals * ioi_in_quarters[time_sig_indices] * inter_onset_ratios
        if len(tempos) > 1 and tempos[-1] > 480.0:
            tempos[-1] = tempos[-2]

        # tempos
        perf_midi_s.tempos = [
            Tempo(time=perf_time, qpm=float(tempo), ttype="second")
            for perf_time, tempo in zip(perf_onsets[:-1], tempos)
        ]
    else:
        perf_midi_s.tempos[0].time = 0

    # process note marker tracks
    if len(marker_tracks) > 0:
        for marker in perf_midi_s.markers:
            marker.time = max(marker.time, 0)
        for track in perf_midi_s.tracks[-len(marker_tracks) :]:
            note_marker = track.name.split()[1]
            for n in track.notes:
                perf_midi_s.markers.append(
                    TextMeta(
                        time=n.time,
                        text=f"{note_marker}_{n.pitch}_{n.time:.6f}_{n.duration:.6f}",
                        ttype="second",
                    )
                )
        perf_midi_s.tracks = perf_midi_s.tracks[: -len(marker_tracks)]

    # convert absolute to symbolic (output MIDI)
    midi = perf_midi_s.to("tick")

    if synchronize_grid:
        # resample if different `ticks_per_quarter` is provided and to avoid zero durations
        resample_midi(midi, ticks_per_quarter, min_duration=1)

        # copy time signatures
        midi.time_signatures = score_midi.time_signatures

    meta = SyncMetadata(
        initial_shift=perf_shift,
        grid_level=str(grid_level),
        num_points=len(score_onsets),
        ticks_per_quarter=score_midi.ticks_per_quarter,
        time_signatures=score_midi.time_signatures,
        ticks_per_bar=ticks_per_bar,
        ibi_in_quarters=ibi_in_quarters,
        tempos=tempos,
    )

    return midi, meta


def shift_by_initial_onset(
    score_midi: Score,
    perf_midi: Score,
    onset_pairs: np.ndarray,
    grid_level: str | GridLevel = GridLevel.BAR,
    ticks_per_quarter: int = 480,
    inplace: bool = True,
) -> Score:
    """Linearly shifts the performance timeline to align with the score's first onset.

    Args:
        score_midi: The score providing the symbolic beat grid.
        perf_midi: The performance to be synchronized.
        onset_pairs: A 2D array of matched (score_tick, perf_time) pairs.
        grid_level: The granularity of the synchronization grid.
            ``GridLevel.BAR`` (or "bar") for measure-level synchronization
            ``GridLevel.BEAT`` (or "beat") for beat-level synchronization.
        ticks_per_quarter: Target MIDI resolution (TPQ). The performance will be resampled to this value.
        inplace: If ``True``, modifies the `perf_midi` object in-place.

    Returns:
        The shifted symusic.Score object. A 'ShiftSync' TextMeta marker is
        added at time 0 to preserve a record of the original timing offset.
    """
    perf_midi = perf_midi if inplace else perf_midi.copy()
    ttype = "tick" if isinstance(perf_midi.ttype, Tick) else "second"

    # get absolute timing of tracks
    perf_midi_s = perf_midi.to("second")

    # compute score and performance onsets
    score_bars, score_beats = get_bar_beat_ticks(score_midi)
    score_onsets = score_bars if grid_level == GridLevel.BAR else score_beats
    score_onsets, perf_onsets = get_performance_beats(
        score_onsets,
        onset_pairs,
        monotonic_times=True,
        ticks_per_quarter=ticks_per_quarter,
    )
    perf_shift = min(perf_onsets[0], perf_midi_s.start())

    # shift onsets and performance events
    perf_midi_s.shift_time(-perf_shift, inplace=True)
    perf_midi_s.markers.insert(
        0, TextMeta(time=0, text=f"ShiftSync_{perf_shift:.6f}", ttype="second")
    )

    # convert absolute to symbolic (output MIDI)
    midi = perf_midi_s.to(ttype)

    return midi
