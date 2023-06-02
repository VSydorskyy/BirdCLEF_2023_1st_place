import argparse
import os
from os.path import join as pjoin

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

RS = 42
SHUFFLE = True


def collect_args():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("df_path", type=str, help="Path to train dataframe")
    parser.add_argument(
        "--save_path",
        type=str,
        help="Path to .npy file to save split",
        default="",
    )
    parser.add_argument(
        "--split_col",
        type=str,
        help="Name of stratification column",
        default="target",
    )
    parser.add_argument(
        "--n_folds", type=int, help="Number of CV folds", default=5
    )

    args = parser.parse_args()

    return args


def stratified_k_fold(X, y, k):
    return StratifiedKFold(n_splits=k, shuffle=SHUFFLE, random_state=RS).split(
        X, y
    )


if __name__ == "__main__":

    args = collect_args()

    print(f"Received args: {args}")

    if args.save_path == "":
        save_path = pjoin(os.path.dirname(args.df_path), "cv_split.npy")
    else:
        save_path = args.save_path

    df = pd.read_csv(args.df_path)

    split = list(stratified_k_fold(X=df, y=df[args.split_col], k=args.n_folds))

    np.save(save_path, split)

    print("Done!")
