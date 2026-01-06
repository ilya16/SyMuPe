from symupe.utils import ExplicitEnum
from .words import extract_main_keyword

DYNAMIC_PREFIX = "dynamic"


class DynamicClasses(ExplicitEnum):
    CONSTANT = "constant"
    INCREASE = "increase"
    DECREASE = "decrease"
    ACCENT = "accent"
    ALL = "all"


CONSTANT_DYNAMIC_KEYS = [
    "pppp",
    "ppp",
    "pp",  # very soft
    ("p", "piano"),  # soft
    "mp",  # medium soft
    "mf",  # medium loud
    ("f", "forte"),  # loud / strong
    "ff",  # very loud
    "fff",
    "ffff"
]

INCREASE_DYNAMIC_KEYS = [
    ("crescendo", "cresc", "cresc.")  # getting louder
]

DECREASE_DYNAMIC_KEYS = [
    ("diminuendo", "dim", "decresc", "dim.", "descresc."),  # getting softer
    ("smorzando", "smorz.")  # dying away
]

ACC_DYNAMIC_KEYS = [
    "fp",
    "ffp",
    "pf",
    ("sf", "fz", "sfz", "sffz", "sforzando", "forzando"),
    ("rf", "rfz", "rinforzando")
]

DYNAMIC_DIRECTIONS_MAPS = {
    DynamicClasses.CONSTANT: CONSTANT_DYNAMIC_KEYS,
    DynamicClasses.INCREASE: INCREASE_DYNAMIC_KEYS,
    DynamicClasses.DECREASE: DECREASE_DYNAMIC_KEYS,
    DynamicClasses.ACCENT: ACC_DYNAMIC_KEYS
}
DYNAMIC_DIRECTIONS_MAPS[DynamicClasses.ALL] = sum(DYNAMIC_DIRECTIONS_MAPS.values(), [])

DYNAMIC_DIRECTIONS = {
    key: [f"{DYNAMIC_PREFIX}/{key}/{extract_main_keyword(value)}" for value in values]
    for key, values in DYNAMIC_DIRECTIONS_MAPS.items()
    if key != DynamicClasses.ALL
}

DYNAMIC_DIRECTION_KEYS = sum(DYNAMIC_DIRECTIONS.values(), [])


def hairpin_word_regularization(word):
    if "decresc" in word:
        word = "diminuendo"
    elif "cresc" in word:
        word = "crescendo"
    elif "dim" in word:
        word = "diminuendo"
    return word
