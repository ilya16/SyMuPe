from __future__ import annotations

from miditok.utils import merge_tracks
from symusic import Score, Tempo, TimeSignature
from symusic.core import Tick

from ..midi import utils as midi_utl


def preprocess_midi(
        midi: Score,
        to_single_track: bool = True,
        sort_events: bool = True,
        clean_duplicates: bool = True,
        cut_overlapped_notes: bool = False,
        clean_short_notes: bool = False,
        filter_extra_events: bool = False,
        pitch_range: tuple[int, int] | None = (21, 109),
        min_tick_shift: int | None = None,
        min_tick_duration: int | None = None,
        min_time_shift: float | None = None,
        min_time_duration: float | None = None,
        max_silence: float | None = None,
        downsample_ticks_per_quarter: int | None = None,
        target_ticks_per_quarter: int | None = None
):
    if len(midi.tracks) == 0:
        return midi

    if len(midi.tracks) > 1 and to_single_track:
        merge_tracks(midi.tracks, effects=True)

    midi.time_signatures = midi.time_signatures or [TimeSignature(0, 4, 4, ttype=midi.ttype)]
    midi.tempos = midi.tempos or [Tempo(0, 120, ttype=midi.ttype)]

    midi.time_signatures[0].time = 0
    midi.tempos[0].time = 0

    if sort_events:
        for track in midi.tracks:
            track.notes, _ = midi_utl.sort_notes(track.notes, order="time")

    if clean_duplicates:
        midi_utl.remove_duplicated_midi_changes(midi)

    if downsample_ticks_per_quarter is not None:
        target_ticks_per_quarter = target_ticks_per_quarter or midi.ticks_per_quarter
        midi = midi_utl.resample_midi(midi, ticks_per_quarter=downsample_ticks_per_quarter)

    if isinstance(midi.ttype, Tick):
        min_duration, min_shift = min_tick_duration, min_tick_shift
    else:
        min_duration, min_shift = min_time_duration, min_time_shift

    if min_duration is not None and not clean_short_notes:
        target_ticks_per_quarter = target_ticks_per_quarter or midi.ticks_per_quarter

    if target_ticks_per_quarter is not None:
        midi = midi_utl.resample_midi(midi, ticks_per_quarter=target_ticks_per_quarter, min_duration=min_duration)

    for track in midi.tracks:
        if clean_duplicates:
            track.notes = midi_utl.remove_duplicated_notes(track.notes)

        if cut_overlapped_notes:
            track.notes = midi_utl.cut_overlapping_notes(track.notes, min_shift=min_shift)
            if clean_duplicates:
                track.notes = midi_utl.remove_duplicated_notes(track.notes)

        if clean_short_notes:
            track.notes = midi_utl.remove_short_notes(
                track.notes, time_division=midi.ticks_per_quarter, min_duration=min_duration
            )

        if pitch_range is not None:
            track.notes = midi_utl.filter_notes_by_pitch_range(track.notes, pitch_range=pitch_range)

    midi.tracks = [track for track in midi.tracks if len(track.notes) > 0]

    if max_silence is not None:
        midi = midi_utl.clip_silence(midi, max_silence=max_silence)

    if filter_extra_events:
        midi = midi_utl.filter_extra_midi_events(midi, sort=sort_events, use_sustain_boundaries=True)

    return midi


