from .embeddings import (
    TupleTransformerEmbeddingsRegistry,
    TupleTransformerEmbeddingsConfig, TupleTransformerEmbeddings,
    MultiSeqTupleTransformerEmbeddingsConfig, MultiSeqTupleTransformerEmbeddings,
    PositionTupleTransformerEmbeddingsConfig, PositionTupleTransformerEmbeddings
)
from .heads import (
    TupleTransformerHeadsRegistry,
    TupleTransformerHeadsConfig,
    TupleTransformerLMHeadConfig, TupleTransformerLMHead,
    TupleTransformerTiedLMHeadConfig, TupleTransformerTiedLMHead,
    TupleTransformerTiedSplitLMHeadConfig, TupleTransformerTiedSplitLMHead,
    TupleTransformerCausalLMHeadConfig, TupleTransformerCausalLMHead,
    TupleTransformerSplitValueHeadConfig, TupleTransformerSplitValueHead,
    TupleTransformerEmbeddingHeadConfig, TupleTransformerEmbeddingHead
)
from .transformer import (
    TupleTransformerCache, TupleTransformerOutput,
    TupleTransformerConfig, TupleTransformer
)
from .language_modeling import (
    TupleTransformerLMWrapper,
    TupleTransformerARWrapper,
    TupleTransformerMLMWrapper,
    TupleTransformerMixedLMWrapper
)
from .flow_matching import (
    TupleTransformerCFMOutput, TupleTransformerCFMWrapper, CFMIntermediates,
    TupleTransformerDFMOutput, TupleTransformerDFMWrapper, DFMIntermediates,
    TupleTransformerFMOutput, TupleTransformerFMWrapper, FMIntermediates
)
from ..classes import LanguageModelingMode, ValuePredictionMode

TupleTransformerWrappers = {
    LanguageModelingMode.MLM: TupleTransformerMLMWrapper,
    LanguageModelingMode.CLM: TupleTransformerARWrapper,
    LanguageModelingMode.MixedLM: TupleTransformerMixedLMWrapper,
    ValuePredictionMode.CFM: TupleTransformerCFMWrapper,
    LanguageModelingMode.DFM: TupleTransformerDFMWrapper,
    LanguageModelingMode.FM: TupleTransformerFMWrapper,
}
