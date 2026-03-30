from symupe.utils import ExplicitEnum
from .words import extract_main_keyword

ARTICULATION_PREFIX = "articulation"


class ArticulationClasses(ExplicitEnum):
    NOTE = "note"
    INTERVAL = "interval"
    ORNAMENT = "ornament"
    SLUR = "slur"
    ALL = "all"


NOTE_ARTICULATION_KEYS = [
    "accent",
    "strong-accent",
    "arpeggiate",
    "staccato",
    "staccatissimo",
    "tenuto",
    ("legato", "legatissimo", "ligato", "ligatissimo"),
    "detached-legato",
    "fermata",
    "spiccato",
    "scoop",
    "plop",
    "doit",
    "falloff",
    "breath-mark",
    "caesura",
    "stress",
    "unstress",
    "soft-accent",
]

INTERVAL_ARTICULATION_KEYS = [
    ("interval/staccato", "staccato"),
    ("interval/legato", "legato", "legatissimo", "ligato", "ligatissimo"),
]

ORNAMENT_KEYS = [
    "trill-mark",
    "turn",
    "wavy-line",
    "mordent",
    "inverted-mordent",
    "tremolo",
    "delayed-turn",
    "inverted-turn",
    "delayed-inverted-turn",
    "vertical-turn",
    "inverted-vertical-turn",
    "shake",
    "schleifer",
    "haydn",
]

SLUR_KEYS = [
    "slur-start",
    "slur-stop",
]

ARTICULATION_DIRECTIONS_MAPS = {
    ArticulationClasses.NOTE: NOTE_ARTICULATION_KEYS,
    ArticulationClasses.INTERVAL: INTERVAL_ARTICULATION_KEYS,
    ArticulationClasses.ORNAMENT: ORNAMENT_KEYS,
    ArticulationClasses.SLUR: SLUR_KEYS,
}
ARTICULATION_DIRECTIONS_MAPS[ArticulationClasses.ALL] = sum(
    ARTICULATION_DIRECTIONS_MAPS.values(), []
)

ARTICULATION_DIRECTIONS = {
    key: [f"{ARTICULATION_PREFIX}/{key}/{extract_main_keyword(value)}" for value in values]
    for key, values in ARTICULATION_DIRECTIONS_MAPS.items()
    if key != ArticulationClasses.ALL
}

ARTICULATION_DIRECTION_KEYS = sum(ARTICULATION_DIRECTIONS.values(), [])
