from __future__ import annotations

import copy
import warnings
from abc import ABC, abstractmethod
from dataclasses import replace, dataclass
from pathlib import Path

import numpy as np
import torch
from symusic import Score

from symupe.data.datasets import SequenceDataset
from symupe.data.tokenizers import SyMuPe, TokSequence, SequenceType
from symupe.models import Model
from .base import Generator


@dataclass
class PerformanceRenderingResult:
    """Storage for performance rendering results.

    Args:
        score_midi: Input quantized score MIDI object.
        score_seq: Input score token sequence.
        gen_seq: Raw generated token sequence.
        perf_seq: Post-processed performance token sequence.
        perf_midi: Generated MIDI score with raw performed durations.
        perf_midi_sus: Generated MIDI score with applied sustain pedals.
    """

    score_midi: Score
    score_seq: TokSequence
    gen_seq: TokSequence
    perf_seq: TokSequence
    perf_midi: Score | None = None
    perf_midi_sus: Score | None = None


class PerformanceGenerator(Generator, ABC):
    """Inference wrapper for score-to-performance generation tasks.

    Provides high-level methods for rendering expressive performances from quantized scores.
    """

    def __init__(
        self,
        model: Model,
        tokenizer: SyMuPe,
        dataset: SequenceDataset | None = None,
        used_token_types: list[str] | None = None,
        mask_token_dims: dict[str, list[int]] | list[int] | None = None,
        used_context_token_types: list[str] | None = None,
        used_score_token_types: list[str] | None = None,
        device: str | torch.device | None = None,
        **kwargs,
    ):
        """Initializes performance generator with model and tokenizer.

        Args:
            model: :class:`Model` instance used for generation.
            tokenizer: :class:`SyMuPe` tokenizer instance for encoding.
            dataset: Optional dataset for sequence extraction.
            used_token_types: List of token types used by generator.
            mask_token_dims: Indices of tokens to mask during generation.
            used_context_token_types: List of token types for context sequence.
            used_score_token_types: List of token types for score sequence.
            device: Target computation device.
            **kwargs: Additional parameters for base generator.
        """
        super().__init__(model, tokenizer, dataset, device, **kwargs)

        cfg = model._config

        self.used_token_types = used_token_types or list(cfg.num_tokens.keys())

        if mask_token_dims is None:
            predicted_token_types = list(cfg.get("token_keys", [])) + list(
                cfg.get("value_keys", [])
            )
            mask_token_dims = {
                "performance": [
                    i for i, t in enumerate(self.used_token_types) if t in predicted_token_types
                ]
            }
        self.mask_token_dims = mask_token_dims or {}

        self.used_context_token_types = (
            used_context_token_types or list((cfg.context_num_tokens or {}).keys()) or None
        )
        self.used_score_token_types = (
            used_score_token_types or list((cfg.score_num_tokens or {}).keys()) or None
        )

    def _tokenize_midi(self, score: Score) -> TokSequence:
        """Converts :class:`symusic.Score` object into a score-based token sequence.

        Args:
            score: Preprocessed :class:`symusic.Score` object.

        Returns:
            Raw :class:`TokSequence` in score encoding.
        """
        score_seq = self.tokenizer.encode_score(score)
        score_seq = self.tokenizer.score_tokens_as_performance(score_seq)
        return score_seq

    def __call__(self, *args, **kwargs) -> list[PerformanceRenderingResult]:
        """Alias for :meth:`perform_score`."""
        return self.perform_score(*args, **kwargs)

    @abstractmethod
    def _prepare_generator_kwargs(self, **kwargs) -> dict[str, object]:
        """Extracts model-specific hyperparameters for generation.

        Args:
            **kwargs: Raw generation parameters.

        Returns:
            Dictionary of processed hyperparameters for model inference.
        """
        return kwargs

    def perform_score(
        self,
        score: str | Path | Score | TokSequence,
        use_score_context: bool = True,
        num_samples: int = 8,
        max_new_notes: int = 64,
        max_seq_len: int = 256,
        cfm_steps: int = 10,
        cfm_step_factor: float = 0.75,
        lm_top_k: float = -1.0,
        lm_top_p: float = 0.95,
        seed: int | None = None,
        show_progress: bool = True,
        **kwargs,
    ) -> list[PerformanceRenderingResult]:
        """Renders expressive performances from musical score.

        Handles MIDI loading, tokenization, batch generation, and musical post-processing.

        Args:
            score: Input MIDI path, :class:`symusic.Score` object, or :class:`TokSequence` object.
            use_score_context: Whether to provide score as context sequence to model.
            num_samples: Number of variations to generate.
            max_new_notes: Maximum notes to generate per window.
            max_seq_len: Maximum sequence length for transformer attention.
            cfm_steps: Number of steps for flow matching.
            cfm_step_factor: Scaling factor for flow matching steps.
            lm_top_k: Top-k filtering threshold for token-based models.
            lm_top_p: Nucleus sampling threshold for token-based models.
            seed: Random seed for reproducibility.
            show_progress: Whether to display tqdm progress bar.
            **kwargs: Additional model-specific inference parameters.

        Returns:
            List of :class:`PerformanceRenderingResult` objects.
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        # load MIDI and prepare token sequence
        if not isinstance(score, TokSequence):
            score = self._load_midi(score)

        score_seq = self._tokenize_midi(score)

        # prepare internal data
        self.reset()
        self.prepare_sequence(
            seq=score_seq,
            task="performance",
            score_seq=score_seq if use_score_context else None,
            add_sos_eos=True,
            context_len=0,
        )

        # prepare model inference kwargs
        generator_kwargs = self._prepare_generator_kwargs(
            cfm_steps=cfm_steps,
            cfm_step_factor=cfm_step_factor,
            lm_top_k=lm_top_k,
            lm_top_p=lm_top_p,
            **kwargs,
        )

        # performance score sequence
        gen_tokens, gen_values = self.generate_batch(
            num_sequences=num_samples,
            max_new_notes=max_new_notes,
            max_seq_len=max_seq_len,
            group_onset_notes=True,
            show_progress=show_progress,
            **generator_kwargs,
        )

        # aggregate results
        results = []
        for i in range(len(gen_tokens)):
            raw_gen_seq = self.tokenizer.decompress(
                self.tokenizer.denormalize_values(
                    replace(
                        self.data.init_seq,
                        ids=gen_tokens[i],
                        values=gen_values[i],
                        type=SequenceType.PERFORMANCE,
                    ).numpy()
                )
            )

            perf_seq = self._postprocess_performance_sequence(
                copy.deepcopy(raw_gen_seq), score_seq=score_seq
            )

            perf_midi, perf_midi_sus = self._performance_sequence_to_midi(perf_seq)

            results.append(
                PerformanceRenderingResult(
                    score_midi=score,
                    score_seq=score_seq,
                    gen_seq=raw_gen_seq,
                    perf_seq=perf_seq,
                    perf_midi=perf_midi,
                    perf_midi_sus=perf_midi_sus,
                )
            )

        return results

    @abstractmethod
    def generate_batch(
        self,
        num_sequences: int = 1,
        max_new_notes: int = 16,
        max_seq_len: int = 512,
        cond_control: torch.Tensor | None = None,
        cond_seq_control: dict[str, float] | None = None,
        interpolated: bool = False,
        group_onset_notes: bool = True,
        show_progress: bool = True,
        **model_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generates batch of sequences using model backbone.

        Args:
            num_sequences: Number of sequences in batch.
            max_new_notes: Maximum notes to generate.
            max_seq_len: Maximum sequence length.
            cond_control: Conditional embedding tensors.
            cond_seq_control: Dictionary of control values for context sequence.
            interpolated: Whether to use interpolated generation.
            group_onset_notes: Whether to group notes with same onset.
            show_progress: Whether to display tqdm progress bar.
            **model_kwargs: Additional arguments for model generate method.

        Returns:
            Tuple of generated token IDs and values.
        """
        ...

    def _postprocess_performance_sequence(
        self,
        gen_seq: TokSequence,
        score_seq: TokSequence,
    ) -> TokSequence:
        """Applies musical metadata and sustain logic to generated sequence.

        Copies `Bar`, `Tempo`, and `Time Signature` information from input score.

        Args:
            gen_seq: Generated TokSequence to process.
            score_seq: Original score TokSequence for metadata reference.

        Returns:
            Post-processed :class:`TokSequence` with musical metadata.
        """
        vocab = gen_seq.vocab

        for key in ["Bar", "Tempo"] + self.tokenizer.time_signature_tokens:
            if self.tokenizer.has_token_types(gen_seq, key):
                continue
            gen_seq.ids[:, vocab[key]] = score_seq.ids[:, vocab[key]]
            gen_seq.values[:, vocab[key]] = score_seq.values[:, vocab[key]]

        if "Sustained" in vocab and not self.tokenizer.has_token_types(gen_seq, "Sustained"):
            if "TimeDurationSustain" in vocab and "TimeDuration" in vocab:
                sustained = (
                    gen_seq.values[:, vocab["TimeDurationSustain"]]
                    > gen_seq.values[:, vocab["TimeDuration"]]
                )

                gen_seq.values[:, vocab["Sustained"]] = sustained
                gen_seq.ids[:, vocab["Sustained"]] = self.tokenizer.zero_token + sustained.astype(
                    int
                )

        return gen_seq

    def _performance_sequence_to_midi(self, gen_seq: TokSequence):
        """Converts token sequence to MIDI score objects.

        Args:
            gen_seq: Token sequence to decode.

        Returns:
            Tuple of MIDI scores (standard, sustained).
        """
        tokenizer = self.tokenizer
        gen_seq = self.token_transformer(copy.deepcopy(gen_seq), "time_performance")

        gen_midi, gen_midi_sus = None, None
        if tokenizer.has_token_types(gen_seq, "TimeDuration"):
            gen_midi = tokenizer.decode_performance(gen_seq, sync_midi=False)

        if "TimeDurationSustain" in gen_seq.vocab:
            gen_seq_sus = replace(gen_seq, values=gen_seq.values.copy(), ids=gen_seq.ids.copy())

            td_idx, tdsus_idx = (
                gen_seq_sus.vocab["TimeDuration"],
                gen_seq_sus.vocab["TimeDurationSustain"],
            )
            note_mask = gen_seq_sus.ids[:, tdsus_idx] >= tokenizer.zero_token
            gen_seq_sus.values[note_mask, td_idx] = gen_seq_sus.values[note_mask, tdsus_idx]
            gen_seq_sus.ids[note_mask, td_idx] = gen_seq_sus.ids[note_mask, tdsus_idx]

            gen_midi_sus = tokenizer.decode_performance(gen_seq_sus, sync_midi=False)

        return gen_midi, gen_midi_sus

    @staticmethod
    def save_performances(
        results: list[PerformanceRenderingResult],
        out_dir: str | Path,
        save_tokens: bool = False,
        save_midi: bool = True,
    ) -> None:
        """Exports rendering results to disk.

        Args:
            results: List of performance rendering results.
            out_dir: Directory to save output files.
            save_tokens: Whether to save compressed token sequences as `.npz`.
            save_midi: Whether to save `.mid` files.
        """
        if not results:
            return

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for i, result in enumerate(results):
            base_name = f"{i + 1:02d}"

            if save_tokens:
                tokens_path = out_dir / f"{base_name}.npz"
                np.savez_compressed(
                    tokens_path,
                    values=result.perf_seq.values,
                    ids=result.perf_seq.ids,
                    type="performance",
                    vocab=result.perf_seq.vocab,
                    meta=result.perf_seq.meta,
                )

            if save_midi:
                if result.perf_midi is not None:
                    result.perf_midi.dump_midi(out_dir / f"{base_name}.mid")

                if result.perf_midi_sus is not None:
                    result.perf_midi_sus.dump_midi(out_dir / f"{base_name}_sus.mid")


def perform_score(
    generator: PerformanceGenerator,
    score: str | Path | Score | TokSequence,
    use_score_context: bool = True,
    num_samples: int = 8,
    max_new_notes: int = 64,
    max_seq_len: int = 256,
    cfm_steps: int = 10,
    cfm_step_factor: float = 0.75,
    lm_top_k: float = -1.0,
    lm_top_p: float = 0.95,
    seed: int | None = None,
) -> list[PerformanceRenderingResult]:
    warnings.warn(
        "`perform_score` is deprecated and will be removed in the future update. "
        "Use `generator.perform_score(score, **kwargs)` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return generator.perform_score(
        score=score,
        use_score_context=use_score_context,
        num_samples=num_samples,
        max_new_notes=max_new_notes,
        max_seq_len=max_seq_len,
        cfm_steps=cfm_steps,
        cfm_step_factor=cfm_step_factor,
        lm_top_k=lm_top_k,
        lm_top_p=lm_top_p,
        seed=seed,
    )


def save_performances(
    results: list[PerformanceRenderingResult],
    out_dir: str | Path,
    save_tokens: bool = False,
    save_midi: bool = True,
) -> None:
    warnings.warn(
        "`save_performances` is deprecated and will be removed in the future update. "
        "Use `generator.save_results(results, **kwargs)` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    PerformanceGenerator.save_performances(
        results=results,
        out_dir=out_dir,
        save_tokens=save_tokens,
        save_midi=save_midi,
    )


def test():
    import torch
    from symusic import Score
    from symupe.inference import AutoGenerator

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build Generator by loading the model and tokenizer directly from the Hub
    generator: PerformanceGenerator = AutoGenerator.from_pretrained(
        "SyMuPe/MLM-base", device=device
    )
    # model, tokenizer = generator.model, generator.tokenizer

    # Load score MIDI
    score_midi = Score("score.mid")

    # Perform score MIDI (tokenization is handled inside)
    gen_results = generator.perform_score(
        score=score_midi,
        use_score_context=True,
        num_samples=8,
    )
    # gen_results[i] is PerformanceRenderingResult(...) containing:
    # - score_midi, score_seq, gen_seq, perf_seq, perf_midi, perf_midi_sus

    # Save performed MIDI files in a single directory
    generator.save_performances(
        gen_results,
        out_dir="samples",
        save_tokens=False,
        save_midi=True,
    )


if __name__ == "__main__":
    test()
