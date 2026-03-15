import torch

from symupe.data.tokenizers import SyMuPe
from symupe.models import (
    Model, Generator,
    MusicTransformer, MusicTransformerGenerator,
    CFMMusicTransformer, CFMMusicTransformerGenerator,
    Seq2SeqMusicTransformer, Seq2SeqMusicTransformerGenerator
)


class AutoGenerator:
    @classmethod
    def from_model(cls, model: Model, tokenizer: SyMuPe, device: torch.device = None, **kwargs) -> Generator:
        """
        Factory to create the correct generator based on the model's architecture.
        """

        # get Generator class from the model
        model_name = model.__class__.__name__
        if isinstance(model, MusicTransformer):
            generator_cls = MusicTransformerGenerator
        elif isinstance(model, CFMMusicTransformer):
            generator_cls = CFMMusicTransformerGenerator
        elif isinstance(model, Seq2SeqMusicTransformer):
            generator_cls = Seq2SeqMusicTransformerGenerator
        else:
            raise ValueError(f"Model of class {model_name} does not have a Generator class")

        # parse model config
        cfg = model._config

        used_token_types = list(cfg.num_tokens.keys())
        predicted_token_types = list(cfg.get("token_keys", [])) + list(cfg.get("value_keys", []))
        mask_token_dims = {
            "performance": [
                i for i, t in enumerate(used_token_types) if t in predicted_token_types
            ]
        }
        used_context_token_types = list((cfg.context_num_tokens or {}).keys()) or None
        used_score_token_types = list((cfg.score_num_tokens or {}).keys()) or None

        return generator_cls(
            model=model,
            tokenizer=tokenizer,
            used_token_types=used_token_types,
            mask_token_dims=mask_token_dims,
            used_context_token_types=used_context_token_types,
            used_score_token_types=used_score_token_types,
            device=device,
            **kwargs
        )
