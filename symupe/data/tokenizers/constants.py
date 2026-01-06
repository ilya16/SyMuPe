""" Tokenizer related constants. """

# override MidiTok default special tokens for backward compatibility
SOS_TOKEN_NAME = "BOS"  # noqa: S105
EOS_TOKEN_NAME = "EOS"  # noqa: S105
SPECIAL_TOKENS = ["PAD", "MASK", SOS_TOKEN_NAME, EOS_TOKEN_NAME]

PAD_TOKEN = "PAD_None"
MASK_TOKEN = "MASK_None"
SOS_TOKEN = f"{SOS_TOKEN_NAME}_None"
EOS_TOKEN = f"{EOS_TOKEN_NAME}_None"
IGNORE_TOKEN = "IGNORE_None"

EOD_TOKEN = "EOD_None"
BAR_LINE_TOKEN = "Bar_Line"
PEDAL_ON_TOKEN = "Pedal_On"
PEDAL_OFF_TOKEN = "Pedal_Off"
TIME_SEGMENT_TOKEN = "Time_Segment"

SPECIAL_TOKENS_VALUE = -100.

TICKS_PER_QUARTER = 480
NOTE_ON_MIDI_EVENT = 144

PLAIN_SCORE_KEYS = [
    "Bar",
    "Position",
    "Pitch",
    "Duration",
    "TimeSig",
    "BeatDuration",
    "BeatsInBar",
    "MaxPosition",
    "Program",
    "PitchClass",
    "PitchOctave",
    "PositionShift",
    "NotesInOnset",
    "PositionInOnset"
]
SCORE_KEYS = PLAIN_SCORE_KEYS + [
    "Velocity",
    "Tempo"
]
PERFORMANCE_KEYS = SCORE_KEYS + [
    "OnsetDev",
    "PerfDuration",
    "RelOnsetDev",
    "RelPerfDuration",
    "TimeShift",
    "TimeDuration",
    "TimePosition",
    "TimeDurationSustain",
    "Sustained"
]
REL_PERFORMANCE_KEYS = SCORE_KEYS + [
    "OnsetDev",
    "PerfDuration",
    "RelOnsetDev",
    "RelPerfDuration",
]
TIME_PERFORMANCE_KEYS = [
    "Pitch",
    "Velocity",
    "TimeShift",
    "TimeDuration",
    "TimePosition",
    "TimeDurationSustain",
    "Sustained",
    "PitchClass",
    "PitchOctave"
]
COMPRESSIBLE_TOKENS = [
    "PitchClass",
    "PitchOctave",
    "PositionShift",
    "NotesInOnset",
    "PositionInOnset",
    "TimePosition"
]
