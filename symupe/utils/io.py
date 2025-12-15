import json
import os
from pathlib import Path

import numpy as np


def load_json(file_path: str | Path) -> dict:
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return {}


def dump_json(data: dict, file_path: str | Path, indent: int | None = None):
    with open(file_path, "w") as f:
        return json.dump(data, f, indent=indent)


def load_npz(file_path: str | Path) -> dict:
    if os.path.exists(file_path):
        return np.load(file_path)
    return {}
