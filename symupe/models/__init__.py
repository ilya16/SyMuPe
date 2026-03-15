from .auto import AutoModel, AutoEvaluator
from .base import Model, Evaluator, Generator
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

for _name, _cls in list(locals().items()):
    if isinstance(_cls, type) and issubclass(_cls, Model) and _cls is not Model:
        AutoModel.register(_cls)

for _name, _cls in list(locals().items()):
    if isinstance(_cls, type) and issubclass(_cls, Evaluator) and _cls is not Evaluator:
        AutoEvaluator.register(_cls)
