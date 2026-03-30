"""
Sources:
    https://www.trinitycollege.com/resource/?id=785
    https://www.music4vip.eu/music_theory_book/en/12
    https://en.wikipedia.org/wiki/Glossary_of_music_terminology
    https://www.absp.org.uk/words/musicdirections.shtml
"""

from symupe.utils import ExplicitEnum
from .words import extract_main_keyword

TEMPO_PREFIX = "tempo"


class TempoClasses(ExplicitEnum):
    CONSTANT = "constant"
    INCREASE = "increase"
    DECREASE = "decrease"
    RESET = "reset"
    ALL = "all"


CONSTANT_TEMPO_KEYS = [
    "grave",  # extremely slow, solemn
    "largo",  # broad / slow
    "larghetto",  # rather broad  / quite slow
    "lento",  # slow
    "adagio",  # slow, leisurely
    "andante",  # going at an easy pace  / medium pace, sometimes described as “at a walking pace”
    "andantino",  # at a moderate pace, but not so slow as / slightly faster than andante
    "moderato",  # moderate speed
    "allegretto",  # rather fast / slightly slower than allegro
    "allegro",  # fast / a brisk tempo (literally “happy”)
    ("vivace", "vivo", "vivacissimamente"),  # lively
    "presto",  # very quick / fast
    "prestissimo",  # very quick indeed, as fast as possible / very fast
    # compound
    "allegro moderato",  # moderately quick
    # expression
    "mosso",  # with motion or animation
    # greater precision / expresion
    "agitato",  # agitated
    "appassionato",  # passionate /impassioned
    "arioso",  # airy, or like an air (a melody) (i.e. in the manner of an aria) / melodious
    "brillante",  # brilliantly, with sparkle / play in a showy and spirited style
    "cantabile",  # singing / in a singing style
    "comodo",  # with ease, relaxed / comfortable (i.e. at moderate speed)  (-)
    "con fuoco",  # with fire  (-)
    "deciso",  # decisive, firm  (-)
    ("dolce", "dolcissimo", "dolciss."),  # sweet
    "delicato",  # delicate  (-)
    ("delicatissimo", "delicatiss", "delicatiss."),  # very delicate
    "dolente",  # sorrowful, plaintive  (-)
    "energico",  # in an energetic, strong manner
    ("espressivo", "espress", "espress."),  # expressively
    "forza",  # force or emphasis  (-)
    "funebre",  # sad, funereal  (-)
    "furioso",  # impetuously, with fury  (-)
    "giocoso",  # playful, humourous  (-)
    "giusto",  # right, exact, strict  # (-)
    "grandioso",  # grandly / in a broad or noble style  (-)
    "grazioso",  # graceful / gracefully, smoothly, elegantly.
    "impetuoso",  # impetuous  (-)
    "langsamer",  # broad tempo / very slow
    (
        "leggiero",
        "legg",
        "leggerissimo",
        "leggermente",
        "leggerezza",
    ),  # light or graceful, delicate and brisk style
    ("lusingando", "lusinghiero"),  # coaxingly, flatteringly, caressingly  (-)
    "maestoso",  # majestic  (-)
    "marcato",  # marked (emphatic manner) / with accentuation, execute every note as if it were to be accented
    "martellato",  # with great force, hammered  # (-)
    "marziale",  # martial, solemn and fierce
    "mesto",  # in a pensive, sad manner
    "morendo",  # dying  (-)
    "pesante",  # heavily, in a ponderous manner
    "piacevole",  # pleasant, playful  (-)
    "pomposo",  # pompous, ceremonious  (-)
    "religioso",  # religious / in a devotional manner  (-)
    "risoluto",  # with emphasis, boldly / in a resolute manner  (-)
    "rubato",  # stolen, robbed (flexible in time)
    ("scherzando", "scherzoso", "scherz", "scherz."),  # playful / in a sprightly, playful manner
    ("secco", "sec"),  # dry (sparse accompaniment, staccato, without resonance)  (-)
    ("semplice", "simplice"),  # simple, plain
    "serioso",  # serious  (-)
    "solenne",  # solemn  (-)
    "sonore",  # sonorous (deep or ringing sound)  (-)
    "sostenuto",  # sustained, lengthened
    ("sotto voce", "sotto"),  # in an undertone (i.e. quietly) / in a subdued manner  (-)
    "teneramente",  # tenderly  (-)
    ("tenuto", "tenute", "ten", "ten."),  # held or sustained
    ("tranquillo", "tranquilo", "tranquilamente"),  # tranquil / calm, peaceful  (-)
    # other
    "menuetto",  # a minuet
    "recitativo",  # recitative (lyrics not to be sung but to be recited, imitating the natural inflections of speech)
]

INCREASE_TEMPO_KEYS = [
    ("accelerando", "acc", "accel", "acc.", "accel."),  # getting (gradually) faster
    (
        "stringendo",
        "string",
        "string.",
    ),  # getting (gradually) faster / pressing onwards, hurrying the speed
    ("più mosso", "piu mosso"),  # quicker at once / getting faster (literally “more moved”)
    "animato",  # animated, lively (faster)
    "stretto",  # getting faster (and often louder as well) / pressing onwards, hurrying the speed
    "slargando",  # becoming (more largo)  (-)
]

DECREASE_TEMPO_KEYS = [
    ("ritardando", "rit", "ritard", "rit.", "ritard."),  # retarding the speed / getting slower
    ("rallentando", "rall", "rall."),  # getting gradually slower
    ("ritenuto", "ritenente", "riten", "riten."),  # held back / getting slower
    "smorzando",  # dying away, getting slower and softer
    "calando",  # softer and slower
    "meno mosso",  # slower at once / getting slower
    "mancando",  # failing or waning tone / dying away  (-)
    "perdendosi",  # losing itself by getting softer and slower
    "slentando",  # becoming slower (more lento)
]

RESET_TEMPO_KEYS = [
    (
        "tempo primo",
        "tempo i",
        "tempo uno",
    ),  # at the same speed as at first / resume the original speed
    ("a tempo", "in tempo"),  # in time
]

TEMPO_DIRECTIONS_MAPS = {
    TempoClasses.CONSTANT: CONSTANT_TEMPO_KEYS,
    TempoClasses.INCREASE: INCREASE_TEMPO_KEYS,
    TempoClasses.DECREASE: DECREASE_TEMPO_KEYS,
    TempoClasses.RESET: RESET_TEMPO_KEYS,
}
TEMPO_DIRECTIONS_MAPS[TempoClasses.ALL] = sum(TEMPO_DIRECTIONS_MAPS.values(), [])

TEMPO_DIRECTIONS = {
    key: [f"{TEMPO_PREFIX}/{key}/{extract_main_keyword(value)}" for value in values]
    for key, values in TEMPO_DIRECTIONS_MAPS.items()
    if key != TempoClasses.ALL
}

TEMPO_DIRECTION_KEYS = sum(TEMPO_DIRECTIONS.values(), [])
