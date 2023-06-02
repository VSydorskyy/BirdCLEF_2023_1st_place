import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def padded_cmap(solution, submission, padding_factor=5):
    solution = solution.drop(["row_id"], axis=1, errors="ignore")
    submission = submission.drop(["row_id"], axis=1, errors="ignore")
    new_rows = []
    for i in range(padding_factor):
        new_rows.append([1 for j in range(len(solution.columns))])
    new_rows = pd.DataFrame(new_rows)
    new_rows.columns = solution.columns
    padded_solution = (
        pd.concat([solution, new_rows]).reset_index(drop=True).copy()
    )
    padded_submission = (
        pd.concat([submission, new_rows]).reset_index(drop=True).copy()
    )
    score = average_precision_score(
        padded_solution.values.astype(int),
        padded_submission.values,
        average="macro",
    )
    return score


def padded_cmap_numpy(y_true, y_pred, padding_factor=5):
    y_true = np.pad(y_true, ((0, padding_factor), (0, 0)), constant_values=1)
    y_pred = np.pad(y_pred, ((0, padding_factor), (0, 0)), constant_values=1)
    return average_precision_score(
        y_true.astype(int),
        y_pred,
        average="macro",
    )
