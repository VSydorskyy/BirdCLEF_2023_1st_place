import json
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel
from tqdm import tqdm


class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def stack_and_max_by_samples(x):
    return np.stack(x).max(axis=0)


def groupby_np_array(groupby_f, array_to_group, apply_f):
    series = (
        pd.DataFrame(
            {
                "groupby_f": groupby_f,
                "array_to_group": [el for el in array_to_group],
            }
        )
        .groupby("groupby_f")["array_to_group"]
        .apply(apply_f)
    )
    return np.stack(series.values)


def load_json(path: str) -> Mapping[str, Any]:
    """
    Read .json file and return dict
    """
    with open(path, "r") as read_file:
        loaded_dict = json.load(read_file)
    return loaded_dict


def write_json(path, data):
    """
    Saves dict into .json file
    """
    with open(path, "w", encoding="utf-8") as f:
        result = json.dump(data, f, ensure_ascii=False, indent=4)
    return result


def load_yaml(path: str) -> Mapping[str, Any]:
    """
    Read .yaml file and return dict
    """
    with open(path, "r") as read_file:
        loaded_dict = yaml.load(read_file, Loader=yaml.FullLoader)
    return loaded_dict
