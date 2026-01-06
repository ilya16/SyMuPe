import numpy as np
from symusic import Score, Track, Tempo, Note, TextMeta
from symusic.core import Tick

from .beats import get_inter_beat_interval, get_bar_beat_ticks, get_performance_beats
from .timing import MIDITimeMapper
from .utils import filter_extra_midi_events, resample_midi
from ..tokenizers.constants import TICKS_PER_QUARTER


def sync_performance_midi(
        score_midi: Score,
        perf_midi: Score,
        onset_pairs: np.ndarray,
        ticks_per_quarter: int = TICKS_PER_QUARTER,
        sync_to_score_grid: bool = True,
        bar_sync: bool = True,
        inplace: bool = True,
        verbose: bool = False
):
    """Synchronize performance MIDI with score MIDI bars/beats through the onset pairs."""
    perf_midi = perf_midi if inplace else perf_midi.copy()
    ttype = "tick" if isinstance(perf_midi.ttype, Tick) else "second"
    time_mapper = MIDITimeMapper(perf_midi) if ttype == "tick" else None

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
            duration = time_mapper.s2t(start0 + duration) - start if time_mapper is not None else duration

            if note_marker not in marker_tracks:
                marker_tracks[note_marker] = Track(program=0, name=f"Marker {note_marker}", ttype=ttype)
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
    score_onsets = score_bars if bar_sync else score_beats
    score_onsets, perf_onsets = get_performance_beats(
        score_onsets, onset_pairs,
        monotonic_times=True, ticks_per_quarter=ticks_per_quarter
    )
    perf_shift = min(perf_onsets[0], min([track.notes.numpy()["time"].min() for track in perf_midi_s.tracks]))

    # shift onsets and performance events
    perf_onsets -= perf_shift
    perf_midi_s.shift_time(-perf_shift, inplace=True)

    perf_midi_s.markers.insert(0, TextMeta(time=0, text=f"ShiftSync_{perf_shift:.6f}", ttype="second"))

    if sync_to_score_grid:
        # align onsets to score grid
        time_sig_ticks, quarter_note_factors, inter_onset_intervals = [], [], []
        for time_sig in score_midi.time_signatures:
            time_sig_ticks.append(time_sig.time)
            quarter_note_factors.append(4 * time_sig.numerator / time_sig.denominator)
            inter_onset_intervals.append(
                get_inter_beat_interval(time_sig=time_sig, ticks_per_quarter=score_midi.ticks_per_quarter)
            )

        time_sig_ticks, quarter_note_factors, inter_onset_intervals = map(
            np.array, (time_sig_ticks, quarter_note_factors, inter_onset_intervals)
        )
        inter_beat_intervals = inter_onset_intervals

        ticks_per_bar = (score_midi.ticks_per_quarter * quarter_note_factors).astype(int)
        beats_per_bar = ticks_per_bar / inter_beat_intervals
        ioi_in_quarters = ibi_in_quarters = quarter_note_factors / beats_per_bar

        if bar_sync:
            inter_onset_intervals = inter_onset_intervals * beats_per_bar
            ioi_in_quarters = ioi_in_quarters * beats_per_bar

        if verbose:
            print(f"score: time_sigs={score_midi.time_signatures}\n"
                  f"       ticks_per_quarter={score_midi.ticks_per_quarter}, ticks_per_bar={ticks_per_bar}\n"
                  f"       inter_beat_intervals={inter_beat_intervals}, inter_onset_intervals={inter_onset_intervals}\n"
                  f"       ibi_in_quarters={ibi_in_quarters}, ioi_in_quarters={ioi_in_quarters}\n"
                  f"perf:  perf_shift={perf_shift}")

        # compute tempos
        intervals = np.diff(perf_onsets)
        if np.any(intervals <= 0.):
            return None

        time_sig_indices = (np.searchsorted(time_sig_ticks, score_onsets, side="right") - 1)[:-1]
        inter_onset_ratios = np.diff(score_onsets) / inter_onset_intervals[time_sig_indices]
        tempos = 60 / intervals * ioi_in_quarters[time_sig_indices] * inter_onset_ratios
        if len(tempos) > 1 and tempos[-1] > 480.:
            tempos[-1] = tempos[-2]

        if verbose:
            print(f"tempos: ({tempos.min():.3f}, {tempos.max():.3f}), {np.median(tempos):.3f}")

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
        for track in perf_midi_s.tracks[-len(marker_tracks):]:
            note_marker = track.name.split()[1]
            for n in track.notes:
                perf_midi_s.markers.append(
                    TextMeta(time=n.time, text=f"{note_marker}_{n.pitch}_{n.time:.6f}_{n.duration:.6f}", ttype="second")
                )
        perf_midi_s.tracks = perf_midi_s.tracks[:-len(marker_tracks)]

    # convert absolute to symbolic (output MIDI)
    midi = perf_midi_s.to("tick")

    if sync_to_score_grid:
        # resample if different `ticks_per_quarter` is provided and to avoid zero durations
        resample_midi(midi, ticks_per_quarter, min_duration=1)

        # copy time signatures
        midi.time_signatures = score_midi.time_signatures

    return midi


def shift_by_initial_onset(
        score_midi: Score,
        perf_midi: Score,
        onset_pairs: np.ndarray,
        ticks_per_quarter: int = TICKS_PER_QUARTER,
        bar_sync: bool = True,
        inplace: bool = True
):
    """Synchronize performance MIDI with score MIDI bars/beats through the onset pairs."""
    perf_midi = perf_midi if inplace else perf_midi.copy()
    ttype = "tick" if isinstance(perf_midi.ttype, Tick) else "second"

    # get absolute timing of tracks
    perf_midi_s = perf_midi.to("second")

    # compute score and performance onsets
    score_bars, score_beats = get_bar_beat_ticks(score_midi)
    score_onsets = score_bars if bar_sync else score_beats
    score_onsets, perf_onsets = get_performance_beats(
        score_onsets, onset_pairs,
        monotonic_times=True, ticks_per_quarter=ticks_per_quarter
    )
    perf_shift = min(perf_onsets[0], perf_midi_s.start())

    # shift onsets and performance events
    perf_midi_s.shift_time(-perf_shift, inplace=True)
    perf_midi_s.markers.insert(0, TextMeta(time=0, text=f"ShiftSync_{perf_shift:.6f}", ttype="second"))

    # convert absolute to symbolic (output MIDI)
    midi = perf_midi_s.to(ttype)

    return midi
