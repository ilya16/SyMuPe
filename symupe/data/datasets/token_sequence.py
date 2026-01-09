""" Token sequence datasets. """
from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path, PurePath

from torch.utils.data import Dataset

from symupe.data.tokenizers import TokSequence
from symupe.utils import apply, load_json
from .utils import cache_to_float


def load_and_process_token_sequence(
        path: str | PurePath,
        load_fn: callable,
        processing_funcs: list[callable] | None = None
):
    seq = load_fn(path)
    if processing_funcs:
        for func in processing_funcs:
            seq = func(seq)
    return seq


class TokenSequenceDataset(Dataset):
    def __init__(self, sequences: list[TokSequence], names: list[str] | None = None):
        self.seqs = sequences

        self.names = names
        if names is not None:
            self._name_to_idx = {name: idx for idx, name in enumerate(self.names)}

    def __getitem__(self, idx: int) -> TokSequence:
        seq = self.seqs[idx]
        return seq[0] if isinstance(seq, tuple) else seq

    def __len__(self) -> int:
        return len(self.seqs)


class LocalTokenSequenceDataset(TokenSequenceDataset):
    def __init__(
            self,
            root: str | PurePath,
            files: list[str | PurePath] | None = None,
            extension: str = ".json",
            load_fn: callable = load_json,
            name_map: callable | None = None,
            preload: bool = False,
            cache: bool | float | int = False  # cache can be a bool or a float ratio in [0, 1]
    ):
        self.root = root
        self.load_fn = load_fn

        if files is None:
            if os.path.isfile(root) and root.lower().endswith(extension):
                files = [Path(root)]
            else:
                files = list(Path(root).glob("**/*" + extension))
            files = list(map(Path, files))
        else:
            files = list(map(lambda x: Path(x).with_suffix(extension), files))

        paths = [PurePath(os.path.join(self.root, file)) for file in files]
        self.paths = paths

        names = [str(file).replace(extension, "") for file in files]
        if name_map is not None:
            names = [name_map(name) for name in names]

        self.cache = cache_to_float(cache, num_files=len(files))

        sequences = self.load_sequences(preload=preload)

        if preload:
            self.max_cache_size = len(sequences)
            self.cache_map = OrderedDict((i, True) for i in range(len(sequences)))
        else:
            self.max_cache_size = int(self.cache * len(self.paths))
            self.cache_map = OrderedDict()

        super().__init__(sequences=sequences, names=names)

    def load_sequence(self, path: str | PurePath):
        return self.load_fn(path)

    def load_sequences(self, preload: bool):
        if preload:
            return apply(self.paths, func=self.load_sequence, desc="Loading token sequences...")
        else:
            return [None] * len(self.paths)

    def __getitem__(self, idx: int) -> TokSequence:
        if self.seqs[idx] is None:
            seq = self.load_sequence(self.paths[idx])
            if self.cache > 0.:
                if len(self.cache_map) > self.max_cache_size:
                    evict_idx, _ = self.cache_map.popitem(last=False)
                    self.seqs[evict_idx] = None

                self.cache_map[idx] = True
                self.seqs[idx] = seq
        else:
            seq = self.seqs[idx]
            # move to most recently used
            self.cache_map.pop(idx)
            self.cache_map[idx] = True

        return seq[0] if isinstance(seq, tuple) else seq
