from symupe.utils import ExplicitEnum
from .classes import (
    SequenceType,
    EncodingType,
    SortingType,
    TokSequence,
    TokSequenceContext,
    TokenizerConfig,
    ENCODING_SORTING,
    SEQUENCE_DEFAULT_ENCODING,
)
from .midi_tokenizer import MusicTokenizer
from .octuple_m import OctupleM
from .symupe import SyMuPe
from .symupe_local import SyMuPeLocal
from .symupe_variants import (
    SyMuPeOnset,
    SyMuPeBeat,
    SyMuPeBar,
    SyMuPeWindow,
    SyMuPeWindowRecompute,
)
from .transformer import SyMuPeTransformer

AutoTokenizer = MusicTokenizer
SyMuPeTokenizer = SyMuPe


class TokenizerType(ExplicitEnum):
    OctupleM = "OctupleM"
    SPMuple = "SPMuple"
    SyMuPe = "SyMuPe"
    SyMuPeLocal = "SyMuPeLocal"
    SyMuPeOnset = "SyMuPeOnset"
    SyMuPeBeat = "SyMuPeBeat"
    SyMuPeBar = "SyMuPeBar"
    SyMuPeWindow = "SyMuPeWindow"
    SyMuPeWindowRecompute = "SyMuPeWindowRecompute"


TOKENIZERS = {
    TokenizerType.OctupleM: OctupleM,
    TokenizerType.SPMuple: SyMuPe,
    TokenizerType.SyMuPe: SyMuPe,
    TokenizerType.SyMuPeLocal: SyMuPeLocal,
    TokenizerType.SyMuPeOnset: SyMuPeOnset,
    TokenizerType.SyMuPeBeat: SyMuPeBeat,
    TokenizerType.SyMuPeBar: SyMuPeBar,
    TokenizerType.SyMuPeWindow: SyMuPeWindow,
    TokenizerType.SyMuPeWindowRecompute: SyMuPeWindowRecompute,
}


__all__ = [
    # classes
    "SequenceType",
    "EncodingType",
    "SortingType",
    "TokSequence",
    "TokSequenceContext",
    "TokenizerConfig",
    "ENCODING_SORTING",
    "SEQUENCE_DEFAULT_ENCODING",
    # tokenizers
    "TokenizerType",
    "TOKENIZERS",
    # core
    "MusicTokenizer",
    "AutoTokenizer",
    # main tokenizers
    "OctupleM",
    "SyMuPe",
    "SyMuPeTokenizer",
    "SyMuPeLocal",
    "SyMuPeTransformer",
    # derivative tokenizers
    "SyMuPeOnset",
    "SyMuPeBeat",
    "SyMuPeBar",
    "SyMuPeWindow",
    "SyMuPeWindowRecompute",
]
