import argparse
import os
from glob import glob
from os.path import join as pjoin

import numpy as np
import pandas as pd

from code_base.utils.audio_utils import parallel_librosa_load


def collect_args():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("au_glob", type=str, help="Glob to find AU files")
    parser.add_argument("save_path", type=str, help="Path to save AU stats")
    parser.add_argument(
        "--n_cores",
        default=32,
        type=int,
        help="Number of cores to use for loading AU files",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = collect_args()

    print(f"Received args: {args}")
    assert args.save_path.endswith(".csv")

    au_paths = glob(args.au_glob, recursive=True)
    print(f"Found {len(au_paths)} AU files")

    stats = parallel_librosa_load(
        au_paths,
        n_cores=args.n_cores,
        sr=None,
        mono=True,
        return_au_len=True,
    )
    stats = pd.DataFrame(
        {
            "au_path": au_paths,
            "au_len": [s[0] for s in stats],
            "sample_rate": [s[1] for s in stats],
            "duration_s": [
                s[0] / s[1] if s[1] is not None and s[0] is not None else None
                for s in stats
            ],
        }
    )
    stats.to_csv(args.save_path, index=False)

    print("Done!")
