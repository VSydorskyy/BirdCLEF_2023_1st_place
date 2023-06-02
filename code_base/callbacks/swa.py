import os
from copy import deepcopy
from os.path import join as pjoin
from typing import Optional

import numpy as np
import torch
from catalyst.dl import Callback, CallbackOrder

from ..utils import get_mode_model


class SWACallback(Callback):
    def __init__(
        self,
        num_of_swa_models: int,
        maximize: bool,
        logging_metric: str,
        verbose: bool,
        loader_key: Optional[str] = None,
    ):
        super().__init__(CallbackOrder.External)
        self.model_checkpoints = [None] * num_of_swa_models
        self.scores = (
            [-np.inf] * num_of_swa_models
            if maximize
            else [np.inf] * num_of_swa_models
        )
        self.best_epochs = [None] * num_of_swa_models

        self.maximize = maximize
        self.logging_metric = logging_metric
        self.verbose = verbose
        self.loader_key = loader_key

    def _save_checkpoint(self, runner):
        os.makedirs(pjoin(runner._logdir, "checkpoints"), exist_ok=True)

        if self.loader_key is None:
            path_name = pjoin(
                runner._logdir,
                "checkpoints",
                f"swa_models_{self.logging_metric}.pt",
            )
        else:
            path_name = pjoin(
                runner._logdir,
                "checkpoints",
                f"swa_models_{self.loader_key}_{self.logging_metric}.pt",
            )
        save_dict = {
            i: (s, m, e)
            for i, (s, m, e) in enumerate(
                zip(self.scores, self.model_checkpoints, self.best_epochs)
            )
        }
        torch.save(save_dict, path_name)

    def _put_state_dict_on_cpu(self, sd):
        sd_copy = deepcopy(sd)
        for k, v in sd_copy.items():
            sd_copy[k] = v.cpu()
        return sd_copy

    def on_epoch_end(self, runner):
        if self.loader_key is None:
            epoch_metric = runner.loader_metrics[self.logging_metric]
        else:
            epoch_metric = runner.epoch_metrics[self.loader_key][
                self.logging_metric
            ]
        epoch_n = runner.epoch_step

        if self.maximize:
            if np.min(self.scores) < epoch_metric:
                min_value_index = np.argmin(self.scores)
                self.scores[min_value_index] = epoch_metric
                self.model_checkpoints[
                    min_value_index
                ] = self._put_state_dict_on_cpu(
                    get_mode_model(runner.model, mode="val").state_dict()
                )
                self.best_epochs[min_value_index] = epoch_n
        else:
            if np.max(self.scores) > epoch_metric:
                max_value_index = np.argmax(self.scores)
                self.scores[max_value_index] = epoch_metric
                self.model_checkpoints[
                    max_value_index
                ] = self._put_state_dict_on_cpu(
                    get_mode_model(runner.model, mode="val").state_dict()
                )
                self.best_epochs[max_value_index] = epoch_n

        if self.verbose:
            if self.loader_key is None:
                msg = "Best models scores by {} : {}".format(
                    self.logging_metric, self.scores
                )
            else:
                msg = "Best models scores from {} by {} : {}".format(
                    self.loader_key, self.logging_metric, self.scores
                )
            print(msg)

        self._save_checkpoint(runner)
