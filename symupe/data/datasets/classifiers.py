""" Classifier predictions datasets. """
from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path, PurePath

import numpy as np
from torch.utils.data import Dataset

from symupe.utils import apply_dict, load_npz
from .utils import cache_to_float


def load_probabilities(path: str | PurePath):
    return load_npz(path)["probs"]


class EmotionPredictionsDataset(Dataset):
    def __init__(
            self,
            predictions: dict[str, np.ndarray],
            labels: dict[int, str] | None = None,
            label_embeddings: np.ndarray | None = None
    ):
        self.predictions = predictions
        self.labels = labels
        self.label_embeddings = label_embeddings

    def __getitem__(self, name: str) -> np.ndarray | None:
        return self.predictions.get(name, None)

    def __len__(self) -> int:
        return len(self.predictions)


class LocalEmotionPredictionsDataset(EmotionPredictionsDataset):
    def __init__(
            self,
            root: str | PurePath,
            files: list[str | PurePath] | None = None,
            extension: str = ".npz",
            suffix: str = "_emo",
            labels_file: str = "labels.txt",
            label_embeddings_file: str = "label_embeddings.npy",
            load_fn: callable = load_probabilities,
            preload: bool = False,
            cache: bool | float | int = False  # cache can be a bool or a float ratio in [0, 1]
    ):
        self.root = root
        self.load_fn = load_fn

        if files is None:
            if os.path.isfile(root) and root.lower().endswith(extension):
                files = [Path(root)]
            else:
                files = list(Path(root).glob("**/*" + suffix + extension))
            files = list(map(Path, files))
        else:
            files = list(map(lambda x: Path(x + suffix).with_suffix(extension), files))

        paths = [PurePath(os.path.join(self.root, file)) for file in files]
        self.paths = paths

        self.names = [str(file).replace(extension, "").replace(suffix, "") for file in files]
        self._name_to_path = {name: path for name, path in zip(self.names, self.paths)}

        # labels
        labels = None
        labels_path = os.path.join(self.root, labels_file)
        if os.path.exists(labels_path):
            with open(labels_path, "r") as f:
                labels = {i: line.strip() for i, line in enumerate(f)}

        # label embeddings
        label_embeddings = None
        label_embeddings_path = os.path.join(self.root, label_embeddings_file)
        if os.path.exists(label_embeddings_path):
            label_embeddings = np.load(label_embeddings_path)

        # predictions
        predictions = self.load_predictions(preload=preload)

        self.cache = cache_to_float(cache, num_files=len(files))

        if preload:
            self.max_cache_size = len(predictions)
            self.cache_map = OrderedDict((name, True) for name in self.names)
        else:
            self.max_cache_size = int(self.cache * len(self.paths))
            self.cache_map = OrderedDict()

        super().__init__(
            predictions=predictions,
            labels=labels,
            label_embeddings=label_embeddings
        )

    def load_prediction(self, path: str | PurePath):
        return self.load_fn(path) if os.path.exists(path) else None

    def load_predictions(self, preload):
        if preload:
            return apply_dict(self.paths, names=self.names, func=self.load_prediction, desc="Loading predictions...")
        else:
            return {name: None for name in self.names}

    def __getitem__(self, name: str) -> np.ndarray | None:
        if name not in self._name_to_path:
            return None

        prediction = self.predictions.get(name, None)
        if prediction is None:
            prediction = self.load_prediction(self._name_to_path[name])
            if self.cache > 0.:
                if len(self.cache_map) > self.max_cache_size:
                    evict_name, _ = self.cache_map.popitem(last=False)
                    self.predictions[evict_name] = None

                self.cache_map[name] = True
                self.predictions[name] = prediction
        else:
            # move to most recently used
            self.cache_map.pop(name)
            self.cache_map[name] = True

        return prediction
