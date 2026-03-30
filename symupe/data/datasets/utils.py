"""Common datasets" utils."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from symupe.utils import find_closest
from ..tokenizers import (
    MusicTokenizer,
    OctupleM,
    SyMuPe,
    SequenceType,
    TokSequence,
    SEQUENCE_DEFAULT_ENCODING,
)


def load_tokens_dict(path: str | Path, tokenizer: MusicTokenizer):
    data = tokenizer.load_tokens(path, raw=True)
    for key in ["ids", "values", "pedals"]:
        data[key] = np.array(data[key]) if data[key] is not None else None
    return data


def load_tokens_npz(path: str | Path):
    data = np.load(path, allow_pickle=True)
    data = {key: data[key] for key in data.files}
    return {key: value.item() if value.ndim == 0 else value for key, value in data.items()}


def _load_tokens(data: dict, tokenizer: MusicTokenizer):
    tok_sequence = TokSequence(
        ids=data.get("ids", None),
        values=data.get("values", None),
        pedals=data.get("pedals", None),
        interpolated=data.get("interpolated", None),
        type=data.get("type", SequenceType.SCORE),
        encoding=data.get(
            "encoding", SEQUENCE_DEFAULT_ENCODING[data.get("type", SequenceType.SCORE)]
        ),
        vocab=data.get("vocab", {}),
        meta=data.get("meta", {}),
    )
    if isinstance(tok_sequence.pedals, np.ndarray) and len(tok_sequence.pedals.shape) == 0:
        tok_sequence.pedals = None

    if (tok_sequence.ids is None or len(tok_sequence.ids) == 0) and tok_sequence.values is not None:
        if isinstance(tokenizer, SyMuPe):
            tok_sequence = tokenizer.decompress(tok_sequence)
        tok_sequence.ids = tokenizer.encode_tokens(tok_sequence.values)

    return tok_sequence


def load_token_sequence(path: str | Path, tokenizer: MusicTokenizer) -> TokSequence | None:
    data = None
    if str(path).endswith(".json"):
        data = load_tokens_dict(path, tokenizer)
    elif str(path).endswith(".npz"):
        data = load_tokens_npz(path)

    if data is not None:
        return _load_tokens(data=data, tokenizer=tokenizer)
    return None


def get_num_bars(seq, tokenizer: MusicTokenizer):
    if isinstance(tokenizer, OctupleM):
        bar_idx = tokenizer.vocab_types_idx["Bar"]
        return int(seq[-1, bar_idx] - tokenizer.zero_token + 1)
    else:
        raise ValueError(f"Unsupported tokenizer: {tokenizer.__class__.__name__}")


def compute_sample_positions(
    seq_num_elements: np.ndarray, sliding_window: int, backward: bool = True
):
    length, sample_positions = 0, []
    for num_elem in seq_num_elements:
        positions = np.arange(
            0,
            num_elem - sliding_window // 2 if num_elem > sliding_window // 2 else num_elem,
            sliding_window,
        )
        if backward:
            back_shift = (
                -sliding_window // 4
                if (num_elem - sliding_window // 2) % sliding_window == 0
                else 0
            )
            backward_positions = np.arange(
                num_elem - sliding_window // 2 - back_shift,
                -1 + sliding_window // 2,
                -sliding_window,
            )
            positions = np.concatenate([positions, backward_positions])

        length += len(positions)
        sample_positions.append(positions)

    sample_ids = np.concatenate([[0], np.cumsum(list(map(len, sample_positions)))[:-1]])
    sample_positions = np.concatenate(sample_positions)

    return length, sample_positions, sample_ids


def get_end_bar(bar_indices, start_bar=0, max_seq_len=512, max_bar=256):
    end_bar = np.where(bar_indices <= bar_indices[start_bar] + max_seq_len)[0][-1] - 1
    return min(max(start_bar, end_bar), start_bar + max_bar - 1)


def cache_to_float(cache: bool | float | int, num_files: int):
    if isinstance(cache, bool):
        cache = 1.0 if cache else 0.0
    elif isinstance(cache, float):
        if not (0 <= cache <= 1):
            raise ValueError("cache ratio must be between 0 and 1")
    elif isinstance(cache, int):
        cache = min(1.0, cache / num_files)
        if not (0 <= cache <= 1):
            raise ValueError("cache must be between 0 and 1")
    else:
        raise TypeError("cache must be a bool or a float")
    return cache


def split_metadata(
    reference_metadata: dict[str, list[str]],
    splits: dict[str, float],
    combine_movements: bool = True,
    seed: int | None = None,
):
    data = {split: dict() for split in splits}
    split_probs = np.array(list(splits.values()))
    cum_probs = np.concatenate([[0.0], np.cumsum(split_probs)])

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    score_performances = list(reference_metadata.items())
    np.random.shuffle(score_performances)
    score_performances = dict(score_performances)

    name_set, points = [], []

    if combine_movements:
        composition_movements = defaultdict(dict)
        for score, performances in score_performances.items():
            composition_movements["/".join(score.split("/")[:2])][score] = performances

        name_set += list(composition_movements.keys())

        for composition, mov_perfs in composition_movements.items():
            points.append(sum(map(len, mov_perfs.values())))
    else:
        for score, performances in score_performances.items():
            points.append(len(performances))

        name_set += list(score_performances.keys())

    points = np.concatenate([[0], np.cumsum(points)])
    cut_indices = find_closest(points, cum_probs * points[-1])

    for split, start, end in zip(splits.keys(), cut_indices[:-1], cut_indices[1:]):
        split_scores = name_set[start:end]
        for score in split_scores:
            if combine_movements:
                for movement, performances in composition_movements[score].items():
                    data[split][movement] = performances
            else:
                data[split][score] = score_performances[score]

    for split in data:
        data[split] = dict(sorted(data[split].items()))

    return data


def split_composer_metadata(
    reference_metadata: dict[str, dict[str, list[str]]],
    splits: dict[str, float],
    combine_movements: bool = True,
    random_split_threshold: int = 10,
    seed: int | None = None,
):
    data = {split: dict() for split in splits}
    split_keys, split_probs = np.array(list(splits.keys())), np.array(list(splits.values()))
    cum_probs = np.concatenate([[0.0], np.cumsum(split_probs)])

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    for composer, score_performances in reference_metadata.items():
        num_performances = sum(map(len, score_performances.values()))

        score_performances = list(score_performances.items())
        np.random.shuffle(score_performances)
        score_performances = dict(score_performances)

        name_set, points = [], []
        if combine_movements:
            composition_movements = defaultdict(dict)
            for score, performances in score_performances.items():
                composition_movements["/".join(score.split("/")[:2])][score] = performances

            name_set = list(composition_movements.keys())

            for composition, mov_perfs in composition_movements.items():
                points.append(sum(map(len, mov_perfs.values())))
        else:
            for score, performances in score_performances.items():
                points.append(len(performances))

            name_set = list(score_performances.keys())

        points = np.concatenate([[0], np.cumsum(points)])

        if num_performances > random_split_threshold:
            cut_indices = find_closest(points, cum_probs * points[-1])

            for split, start, end in zip(splits.keys(), cut_indices[:-1], cut_indices[1:]):
                split_scores = name_set[start:end]
                for score in split_scores:
                    if combine_movements:
                        for movement, performances in composition_movements[score].items():
                            data[split][movement] = performances
                    else:
                        data[split][score] = score_performances[score]
        elif combine_movements:
            for composition, mov_perfs in composition_movements.items():
                _split = np.random.choice(split_keys, p=split_probs)
                for movement, performances in mov_perfs.items():
                    data[_split][movement] = performances
        else:
            for score, performances in score_performances.items():
                _split = np.random.choice(split_keys, p=split_probs)
                data[_split][score] = performances

    for split in data:
        data[split] = dict(sorted(data[split].items()))

    return data
