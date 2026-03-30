from __future__ import annotations

import copy
from dataclasses import replace, dataclass
from pathlib import Path

import numpy as np
import torch
from symusic import Score

from symupe.data.midi import preprocess_midi
from symupe.data.tokenizers import SyMuPe, TokSequence, SequenceType
from symupe.models import Generator, CFMMusicTransformerGenerator


@dataclass
class PerformanceRenderingResult:
    score_midi: Score
    score_seq: TokSequence
    gen_seq: TokSequence
    perf_seq: TokSequence
    perf_midi: Score | None = None
    perf_midi_sus: Score | None = None


def perform_score(
    generator: Generator,
    score: Score | TokSequence,
    use_score_context: bool = True,
    num_samples: int = 8,
    max_new_notes: int = 64,
    max_seq_len: int = 256,
    cfm_steps: int = 10,
    cfm_step_factor: float = 0.75,
    mlm_top_k: float = -1.0,
    mlm_top_p: float = 0.95,
    seed: int | None = None,
):
    tokenizer = generator.tokenizer

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    if isinstance(score, Score):
        score = preprocess_midi(score, to_single_track=True)

        score_seq = tokenizer.encode_score(score)
        score_seq = tokenizer.score_tokens_as_performance(score_seq)
    else:
        score_seq = score

    generator.reset()
    generator.prepare_sequence(
        seq=score_seq,
        task="performance",
        score_seq=score_seq if use_score_context else None,
        add_sos_eos=True,
        context_len=0,
    )

    if isinstance(generator, CFMMusicTransformerGenerator):
        generator_kwargs = {
            "steps": cfm_steps,
            "step_factor": cfm_step_factor,
            "method": "euler",
        }
    else:
        generator_kwargs = {
            "top_k": mlm_top_k,
            "top_p": mlm_top_p,
            "filter_key_ids": {
                key: list(range(tokenizer.zero_token)) for key in generator.used_token_types
            },
        }

    gen_tokens, gen_values = generator.generate_batch(
        num_sequences=num_samples,
        max_new_notes=max_new_notes,
        max_seq_len=max_seq_len,
        group_onset_notes=True,
        **generator_kwargs,
    )

    results = []
    for i in range(len(gen_tokens)):
        raw_gen_seq = tokenizer.decompress(
            tokenizer.denormalize_values(
                replace(
                    generator.data.init_seq,
                    ids=gen_tokens[i],
                    values=gen_values[i],
                    type=SequenceType.PERFORMANCE,
                ).numpy()
            )
        )

        perf_seq = postprocess_performance_sequence(
            tokenizer, copy.deepcopy(raw_gen_seq), score_seq=score_seq
        )

        perf_midi, perf_midi_sus = performance_sequence_to_midi(generator, perf_seq)

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


def postprocess_performance_sequence(
    tokenizer: SyMuPe, gen_seq: TokSequence, score_seq: TokSequence
):
    vocab = gen_seq.vocab

    for key in ["Bar", "Tempo"] + tokenizer.time_signature_tokens:
        if tokenizer.has_token_types(gen_seq, key):
            continue
        gen_seq.ids[:, vocab[key]] = score_seq.ids[:, vocab[key]]
        gen_seq.values[:, vocab[key]] = score_seq.values[:, vocab[key]]

    if "Sustained" in vocab and not tokenizer.has_token_types(gen_seq, "Sustained"):
        if "TimeDurationSustain" in vocab and "TimeDuration" in vocab:
            sustained = (
                gen_seq.values[:, vocab["TimeDurationSustain"]]
                > gen_seq.values[:, vocab["TimeDuration"]]
            )

            gen_seq.values[:, vocab["Sustained"]] = sustained
            gen_seq.ids[:, vocab["Sustained"]] = tokenizer.zero_token + sustained.astype(int)

    return gen_seq


def performance_sequence_to_midi(generator: Generator, gen_seq: TokSequence):
    tokenizer = generator.tokenizer
    gen_seq = generator.token_transformer(copy.deepcopy(gen_seq), "time_performance")

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


def save_performances(
    results: list[PerformanceRenderingResult],
    out_dir: str | Path,
    save_tokens: bool = False,
    save_midi: bool = True,
) -> None:
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


if __name__ == "__main__":
    from symusic import Score
    from symupe.models import AutoModel
    from symupe.inference import AutoGenerator

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model and tokenizer directly from the Hub
    model = AutoModel.from_pretrained("SyMuPe/PianoFlow-base").to(device)
    tokenizer = SyMuPe.from_pretrained("SyMuPe/PianoFlow-base")

    # Prepare generator for the model
    generator = AutoGenerator.from_model(model, tokenizer, device=device)

    # Load score MIDI
    score_midi = Score("score.mid")

    # Perform score MIDI (tokenization is handled inside)
    gen_results = perform_score(
        generator=generator,
        score=score_midi,
        use_score_context=True,
        num_samples=8,
    )
    # gen_results[i] is GenerationResult(score_midi=Score(...), score_seq=TokSequence(...), perf_seq=TokSequence(...), perf_midi=Score(...), perf_midi=Score(...))

    # Save performed MIDI files in a single directory
    save_performances(
        gen_results,
        out_dir="samples",
        save_tokens=False,
        save_midi=True,
    )
