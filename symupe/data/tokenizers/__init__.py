from symupe.utils import ExplicitEnum
from .classes import (
    SequenceType,
    EncodingType,
    SortingType,
    TokSequence,
    TokSequenceContext,
    TokenizerConfig,
    ENCODING_SORTING,
    SEQUENCE_DEFAULT_ENCODING
)
from .common import OctupleM
from .midi_tokenizer import MusicTokenizer
from .symupe import (
    SyMuPeBase,
    SyMuPe,
    SyMuPeLocal,

    SyMuPeOnset,
    SyMuPeBeat,
    SyMuPeBar,
    SyMuPeWindow,
    SyMuPeWindowRecompute,

    SyMuPeTransformer
)


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
    TokenizerType.SPMuple: SyMuPeLocal,
    TokenizerType.SyMuPe: SyMuPe,
    TokenizerType.SyMuPeLocal: SyMuPeLocal,
    TokenizerType.SyMuPeOnset: SyMuPeOnset,
    TokenizerType.SyMuPeBeat: SyMuPeBeat,
    TokenizerType.SyMuPeBar: SyMuPeBar,
    TokenizerType.SyMuPeWindow: SyMuPeWindow,
    TokenizerType.SyMuPeWindowRecompute: SyMuPeWindowRecompute
}
