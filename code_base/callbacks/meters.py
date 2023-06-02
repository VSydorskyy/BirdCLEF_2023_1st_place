import json
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
from catalyst.dl import Callback, CallbackOrder
from scipy.special import expit
from sklearn.metrics import f1_score

from ..utils import (
    groupby_np_array,
    padded_cmap_numpy,
    stack_and_max_by_samples,
)


class PaddedCMAPScore(Callback):
    def __init__(
        self,
        metric_name: str = "padded_cmap",
        output_key: str = "logit",
        input_key: str = "target",
        loader_names: Tuple = ("valid"),
        use_sigmoid: bool = True,
        use_timewise_avarage: bool = False,
        aggr_key: Optional[str] = None,
        output_long_key: Optional[str] = None,
        verbose: bool = True,
        label_str2int_mapping_path: Optional[str] = None,
        scored_bird_path: Optional[str] = None,
    ):
        super().__init__(CallbackOrder.Metric)
        self.metric_name = metric_name
        self.loader_names = loader_names

        self.input_key = input_key
        self.output_key = output_key

        self.running_preds = []
        self.running_targets = []

        self.use_sigmoid = use_sigmoid
        self.use_timewise_avarage = use_timewise_avarage
        self.verbose = verbose

        if (
            label_str2int_mapping_path is not None
            and scored_bird_path is not None
        ):
            print("PaddedCMAPScore will be computed on subset of classes")
            label_str2int = json.load(open(label_str2int_mapping_path))
            scored_bird = json.load(open(scored_bird_path))
            self.scored_bird_ids = [label_str2int[el] for el in scored_bird]
        else:
            self.scored_bird_ids = None

        if aggr_key is not None:
            self.aggr_key = aggr_key
            self.running_aggr = []
        else:
            self.aggr_key = None

        if output_long_key is not None:
            self.output_long_key = output_long_key
            self.running_preds_long = []
        else:
            self.output_long_key = None

    def on_batch_end(self, runner):
        if runner.loader_key in self.loader_names:
            y_hat = runner.output[self.output_key]
            if self.use_sigmoid:
                y_hat = torch.sigmoid(y_hat)
            if self.use_timewise_avarage:
                y_hat = y_hat.max(axis=1)[0]
            y_hat = y_hat.detach().cpu().numpy()
            self.running_preds.append(y_hat)

            y = runner.input[self.input_key].detach().cpu().numpy()
            self.running_targets.append(y)

            if self.aggr_key is not None:
                aggr = runner.input[self.aggr_key].detach().cpu().numpy()
                self.running_aggr.append(aggr)

            if self.output_long_key is not None:
                output_long = runner.output[self.output_long_key]
                if self.use_sigmoid:
                    output_long = torch.sigmoid(output_long)
                if self.use_timewise_avarage:
                    output_long = output_long.max(axis=1)[0]
                output_long = output_long.detach().cpu().numpy()
                self.running_preds_long.append(output_long)

    def _print_v(self, msg):
        if self.verbose:
            print(msg)

    def on_loader_end(self, runner):
        if runner.loader_key in self.loader_names:
            y_true = np.concatenate(self.running_targets)
            y_pred = np.concatenate(self.running_preds)
            if self.scored_bird_ids is not None:
                y_true = y_true[:, self.scored_bird_ids]
                y_pred = y_pred[:, self.scored_bird_ids]
            if self.output_long_key is None:
                y_pred_long = None
            else:
                y_pred_long = np.concatenate(self.running_preds_long)
                if self.scored_bird_ids is not None:
                    y_pred_long = y_pred_long[:, self.scored_bird_ids]
            if self.aggr_key is not None:
                y_aggr = np.concatenate(self.running_aggr)
                y_true = groupby_np_array(
                    groupby_f=y_aggr,
                    array_to_group=y_true,
                    apply_f=stack_and_max_by_samples,
                )
                y_pred = groupby_np_array(
                    groupby_f=y_aggr,
                    array_to_group=y_pred,
                    apply_f=stack_and_max_by_samples,
                )
                if y_pred_long is not None:
                    y_pred_long = groupby_np_array(
                        groupby_f=y_aggr,
                        array_to_group=y_pred_long,
                        apply_f=stack_and_max_by_samples,
                    )

            if y_pred_long is None:
                runner.loader_metrics[self.metric_name] = padded_cmap_numpy(
                    y_true=y_true, y_pred=y_pred
                )
            else:
                raise ValueError("y_pred_long is not supported fro now")

            self.running_preds = []
            self.running_targets = []
            if self.aggr_key is not None:
                self.running_aggr = []
            if self.output_long_key is not None:
                self.running_preds_long = []
