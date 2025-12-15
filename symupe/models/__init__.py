from .base import Model
from .classifiers import (
    SequenceClassifier, SequenceClassifierEvaluator
)
from .music_transformer import (
    MusicTransformer, MusicTransformerEvaluator, MusicTransformerGenerator,
    CFMMusicTransformer, CFMMusicTransformerEvaluator, CFMMusicTransformerGenerator,
    DFMMusicTransformer,
    FMMusicTransformer, FMMusicTransformerEvaluator
)
from .pianoflow import (
    PianoFlow, PianoFlowEvaluator, PianoFlowGenerator
)
from .scoreperformer import (
    ScorePerformer, ScorePerformerEvaluator,
    ScorePerformerGenerator, ScorePerformerInpainter
)
from .seq2seq import (
    Seq2SeqMusicTransformer, Seq2SeqMusicTransformerEvaluator, Seq2SeqMusicTransformerGenerator,
    Seq2SeqDFMMusicTransformer,
    Seq2SeqFMMusicTransformer, Seq2SeqFMMusicTransformerEvaluator
)

MODELS = {name: cls for name, cls in globals().items() if ".model." in str(cls)}
EVALUATORS = {name: cls for name, cls in globals().items() if ".evaluator." in str(cls)}
