from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import torch
from symusic import Score

from symupe.data.datasets import SequenceDataset
from symupe.data.tokenizers import SyMuPe, TokSequence
from symupe.models import Model
from .base import Embedder, _PerformanceInference, _SlidingWindowInference


@dataclass
class MusicEmbeddingResult:
    """Storage for musical sequence embedding results.

    Args:
        midi: Original input MIDI object.
        seq: Full token sequence before windowing.
        embeddings: Contextualized hidden states from the target layer.
            Shape: (num_windows, max_seq_len, hidden_size).
        memory_tokens: Processed memory/global tokens if supported by model.
            Shape: (num_windows, num_memory_tokens, hidden_size).
        token_embeddings: Raw note embeddings before transformer processing.
            Shape: (num_windows, max_seq_len, hidden_size).
        hidden_states: Optional list of hidden states for all layers.
            Each tensor shape: (num_windows, max_seq_len, hidden_size).
        sequences: List of windowed :class:`TokSequence` objects processed by the model.
        window_indices: List of (start, end) token indices for each window.
    """

    midi: Score
    seq: TokSequence
    embeddings: torch.Tensor
    memory_tokens: torch.Tensor | None
    token_embeddings: torch.Tensor | None
    hidden_states: list[torch.Tensor] | None
    sequences: list[TokSequence]
    window_indices: list[tuple[int, int]]


class MusicEmbedder(_PerformanceInference, _SlidingWindowInference, Embedder):
    """Inference wrapper for computing embeddings from musical sequences.

    Provides a high-level API for obtaining rich musical representations from
    expressive MIDI data using transformer backbones (e.g., MLM or CLM).
    """

    def __init__(
        self,
        model: Model,
        tokenizer: SyMuPe,
        dataset: SequenceDataset | None = None,
        used_token_types: list[str] | None = None,
        device: str | torch.device | None = None,
        **kwargs,
    ):
        """Initializes music embedder with model and tokenizer.

        Args:
            model: :class:`Model` instance used for feature embedding.
            tokenizer: :class:`SyMuPe` tokenizer instance for encoding.
            dataset: Optional dataset for sequence extraction.
            used_token_types: List of token types used by the backbone.
            device: Target device for computation.
            **kwargs: Additional embedder parameters.
        """
        used_token_types = used_token_types or list(model._config.num_tokens.keys())

        super().__init__(model, tokenizer, dataset, used_token_types, device, **kwargs)

    def _forward_batch(
        self,
        tokens: torch.Tensor,
        values: torch.Tensor,
        layer: int = -1,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, list[torch.Tensor]]:
        """Internal method to perform batch inference on transformer backbone.

        Args:
            tokens: Batch of token IDs.
            values: Batch of token values.
            layer: Target transformer layer index.
            **kwargs: Additional arguments for the model's forward pass.

        Returns:
            Tuple containing normalized embeddings, memory states, raw token embeddings,
            and all layer hidden states.
        """
        out = self.model.unwrapped_transformer(
            tokens,
            values=values,
            output_layer=layer,
            return_cache=True,
            normalized_embeddings=True,
            **kwargs,
        )

        embeddings = out.hidden_state.cpu()
        memory_tokens = out.memory_state.cpu() if out.memory_state is not None else None
        token_embeddings = out.cache.token_emb.cpu()
        hidden_states = [
            layer_cache.output.cpu() for layer_cache in out.cache.transformer.layers
        ]  # unnormalized layer outputs, include memory tokens if available

        return embeddings, memory_tokens, token_embeddings, hidden_states

    @torch.inference_mode()
    def embed(
        self,
        midi: str | Path | Score | TokSequence,
        max_seq_len: int = 512,
        hop_size: int = 256,
        batch_size: int = 16,
        layer: int | None = -1,
        return_hidden_states: bool = False,
        show_progress: bool = True,
    ) -> MusicEmbeddingResult:
        """Computes musical embeddings using a sliding window approach.

        Handles MIDI tokenization, sequence preparation, and windowed inference.
        Returns results as 3D tensors to preserve positional context.

        Args:
            midi: Input MIDI path, :class:`symusic.Score`, or :class:`TokSequence`.
            max_seq_len: Maximum token length for each window.
            hop_size: Number of tokens to shift between windows.
            batch_size: Number of windows processed in parallel batch.
            layer: Target transformer layer. Zero for the outputs of the 0-indexed layer,
                default -1 is the final layer.
            return_hidden_states: Whether to extract all intermediate layers.
            show_progress: Whether to display tqdm progress bar.

        Returns:
            :class:`MusicEmbeddingResult` containing computed representations.
        """
        # load MIDI and prepare token sequence
        if not isinstance(midi, TokSequence):
            midi = self._load_midi(midi)

        seq = self._tokenize_midi(midi)

        init_seq = copy.deepcopy(seq)

        all_outputs, sequences, window_indices = self._sliding_window_inference(
            seq=seq,
            max_seq_len=max_seq_len,
            hop_size=hop_size,
            batch_size=batch_size,
            layer=layer,
            show_progress=show_progress,
        )

        window_embeddings = torch.cat([out[0] for out in all_outputs], dim=0)
        window_token_embeddings = torch.cat([out[2] for out in all_outputs], dim=0)

        window_memory_tokens = None
        if all_outputs[0][1] is not None:
            window_memory_tokens = torch.cat([out[1] for out in all_outputs], dim=0)

        window_hidden_states = None
        if return_hidden_states:
            num_layers = len(all_outputs[0][3])
            window_hidden_states = [
                torch.cat([out[3][layer] for out in all_outputs]) for layer in range(num_layers)
            ]

        return MusicEmbeddingResult(
            midi=midi,
            seq=init_seq,
            embeddings=window_embeddings,
            memory_tokens=window_memory_tokens,
            token_embeddings=window_token_embeddings,
            hidden_states=window_hidden_states,
            sequences=sequences,
            window_indices=window_indices,
        )


def test():
    import torch
    from symusic import Score
    from symupe.inference import AutoEmbedder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build Embedder by loading the model and tokenizer directly from the Hub
    embedder: MusicEmbedder = AutoEmbedder.from_pretrained("SyMuPe/Aria-MIDI-MLM", device=device)
    # model, tokenizer = embedder.model, embedder.tokenizer

    # Load MIDI
    midi = Score("performance.mid")

    # Compute embeddings (tokenization is handled inside)
    result = embedder(midi, max_seq_len=512, hop_size=256, layer=-1)
    # result is MusicEmbeddingResult(...) containing:
    # - midi, seq, embeddings, memory_tokens, token_embeddings, hidden_states, sequences and window_indices
    print(result.embeddings.shape)


if __name__ == "__main__":
    test()
