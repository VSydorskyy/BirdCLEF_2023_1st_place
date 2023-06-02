import torch
import torch.nn as nn

from ..losses.combined_losses import BCEFocal2WayLoss
from ..losses.focal_loss import FocalLoss, FocalLossBCE
from ..utils import get_mode_model

EPSILON_FP16 = 1e-5


class MultilabelClsForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_f = nn.BCEWithLogitsLoss()

    def forward(self, runner, batch):

        mode = "train" if runner.loader_key == "train" else "val"
        model = get_mode_model(runner.model, mode=mode)

        if runner.loader_key == "infer":
            wave, labels, dfidx, _, _ = batch
            inputs = {"target": labels, "dfidx": dfidx}
        else:
            wave, labels = batch
            inputs = {"target": labels}
        logits = model(wave)

        losses = {"loss": self.loss_f(logits, labels)}
        output = {"logit": logits}

        return losses, inputs, output


class MultilabelClsForwardLongShort(nn.Module):
    def __init__(
        self,
        loss_type="baseline",
        framewise_pred_coef=0.5,
        binirize_labels=False,
        use_weights=False,
        class_sum=False,
        use_slected_indices=False,
        long_files_loaders=("infer",),
        batch_aug=None,
        is_output_dict=True,
        use_focal_loss=False,
        use_bce_focal_loss=False,
        config_2way_loss={
            "weights": [1, 1],
            "clipwise_name": "clipwise_logits_long",
            "framewise_name": "framewise_logits_long",
        },
    ):
        super().__init__()
        loss_type in [
            "bseline",
            "prob_baseline",
            "logit_clip_and_max_frame",
            "baseline_and_max_frame",
        ]
        if use_slected_indices:
            if not (class_sum or use_weights):
                raise ValueError(
                    "Selected indices are supported only with `class_sum` OR `use_weights`"
                )
        if use_focal_loss and use_bce_focal_loss:
            raise ValueError(
                "use_focal_loss and use_bce_focal_loss are mutually exclusive"
            )
        self.class_sum = class_sum
        self.use_weights = use_weights
        self.use_slected_indices = use_slected_indices
        self.loss_type = loss_type
        self.framewise_pred_coef = framewise_pred_coef
        self.binirize_labels = binirize_labels
        self.long_files_loaders = long_files_loaders
        self.is_output_dict = is_output_dict
        if not self.is_output_dict:
            if self.loss_type in [
                "baseline_and_max_frame",
                "logit_clip_and_max_frame",
            ]:
                raise ValueError(
                    "Output dict is required for `baseline_and_max_frame` or `logit_clip_and_max_frame`"
                )
        if loss_type == "prob_baseline":
            assert not use_focal_loss and not use_bce_focal_loss
            self.loss_f = nn.BCELoss(
                reduction="none"
                if (self.use_weights or self.class_sum)
                else "mean"
            )
        elif loss_type == "BCEFocal2WayLoss":
            self.loss_f = BCEFocal2WayLoss(
                **config_2way_loss,
            )
        else:
            if use_focal_loss:
                self.loss_f = FocalLoss(
                    reduction="none"
                    if (self.use_weights or self.class_sum)
                    else "mean"
                )
            elif use_bce_focal_loss:
                self.loss_f = FocalLossBCE(
                    reduction="none"
                    if (self.use_weights or self.class_sum)
                    else "mean"
                )
            else:
                self.loss_f = nn.BCEWithLogitsLoss(
                    reduction="none"
                    if (self.use_weights or self.class_sum)
                    else "mean"
                )
        if self.use_weights:
            self.weights = None
        if self.use_slected_indices:
            self.selected_indices = None
        self.batch_aug = batch_aug
        self.batch_aug_device_is_set = False

    def set_weights(self, weights, device=None):
        self.weights = torch.FloatTensor(weights)[None, :]
        if device is not None:
            self.weights = self.weights.to(device)

    def set_selected_indices(self, selected_indices, device=None):
        self.selected_indices = torch.LongTensor(selected_indices)
        if device is not None:
            self.selected_indices = self.selected_indices.to(device)
        if self.use_weights:
            self.weights = self.weights[:, self.selected_indices]

    def forward(self, runner, batch):

        if self.use_weights and self.weights is None:
            raise RuntimeError("Set weights before calling `forward`")

        if self.use_slected_indices and self.selected_indices is None:
            raise RuntimeError("Set selected_indices before calling `forward`")

        mode = "train" if runner.loader_key == "train" else "val"
        model = get_mode_model(runner.model, mode=mode)

        if runner.loader_key in self.long_files_loaders:
            wave, labels, dfidx, _, _ = batch
            inputs = {
                "target": labels,
                "dfidx": dfidx,
            }
        else:
            wave, labels = batch
            if runner.is_train_loader and self.batch_aug is not None:
                if not self.batch_aug_device_is_set:
                    self.batch_aug.to(wave.device)
                    self.batch_aug_device_is_set = True
                wave = self.batch_aug(wave)
            if self.binirize_labels:
                inputs = {"target": (labels > 0).float()}
            else:
                inputs = {"target": labels}

        output = model(wave)

        if self.loss_type == "baseline":
            if self.is_output_dict:
                loss_v = self.loss_f(output["clipwise_logits_long"], labels)
            else:
                loss_v = self.loss_f(output, labels)
        elif self.loss_type == "baseline_and_max_frame":
            loss_v = (
                self.loss_f(output["clipwise_logits_long"], labels)
                * (1 - self.framewise_pred_coef)
                + self.loss_f(
                    output["framewise_logits_long"].max(1)[0], labels
                )
                * self.framewise_pred_coef
            )
        elif self.loss_type == "logit_clip_and_max_frame":
            loss_v = (
                self.loss_f(torch.logit(output["clipwise_pred_long"]), labels)
                * (1 - self.framewise_pred_coef)
                + self.loss_f(
                    output["framewise_logits_long"].max(1)[0], labels
                )
                * self.framewise_pred_coef
            )
        elif self.loss_type == "prob_baseline":
            if self.is_output_dict:
                loss_v = self.loss_f(output["clipwise_pred_long"], labels)
            else:
                loss_v = self.loss_f(output, labels)
        elif self.loss_type == "BCEFocal2WayLoss":
            loss_v = self.loss_f(output, labels)

        if self.use_slected_indices:
            loss_v = loss_v[:, self.selected_indices]

        if self.use_weights:
            loss_v = (self.weights * loss_v).sum(dim=1).mean()
        elif self.class_sum:
            loss_v = loss_v.sum(dim=1).mean()

        losses = {"loss": loss_v}
        if not self.is_output_dict:
            output = {"pred": output}

        return losses, inputs, output


class MultilabelTDNNForward(nn.Module):
    def __init__(
        self,
        loss_f_name="bce",
        binirize_labels=False,
        use_weights=False,
        class_sum=False,
        use_slected_indices=False,
    ):
        super().__init__()
        if use_slected_indices:
            if not (class_sum or use_weights):
                raise ValueError(
                    "Selected indices are supported only with `class_sum` OR `use_weights`"
                )
        loss_f_name in ["bce", "soft_metric"]
        if use_weights and loss_f_name != "bce":
            raise ValueError("Weights supported only for `bce` loss_f_name")
        if loss_f_name == "soft_metric":
            if self.class_sum:
                raise ValueError(
                    "Forward does not support `soft_metric` and `class_sum`"
                )
        self.class_sum = class_sum
        self.use_weights = use_weights
        self.use_slected_indices = use_slected_indices
        self.binirize_labels = binirize_labels
        if loss_f_name == "soft_metric":
            self.loss_f = CompSoftMetric(logits_input=True)
        elif loss_f_name == "bce":
            self.loss_f = nn.BCEWithLogitsLoss(
                reduction="none"
                if (self.use_weights or self.class_sum)
                else "mean"
            )
        if self.use_weights:
            self.weights = None
        if self.use_slected_indices:
            self.selected_indices = None

    def set_weights(self, weights, device=None):
        self.weights = torch.FloatTensor(weights)[None, :]
        if device is not None:
            self.weights = self.weights.to(device)

    def set_selected_indices(self, selected_indices, device=None):
        self.selected_indices = torch.LongTensor(selected_indices)
        if device is not None:
            self.selected_indices = self.selected_indices.to(device)
        if self.use_weights:
            self.weights = self.weights[:, self.selected_indices]

    def forward(self, runner, batch):

        if self.use_weights and self.weights is None:
            raise RuntimeError("Set weights before calling `forward`")

        if self.use_slected_indices and self.selected_indices is None:
            raise RuntimeError("Set selected_indices before calling `forward`")

        mode = "train" if runner.loader_key == "train" else "val"
        model = get_mode_model(runner.model, mode=mode)

        if runner.loader_key == "infer":
            wave, labels, dfidx, _, _ = batch
            inputs = {
                "target": labels,
                "dfidx": dfidx,
            }
        else:
            wave, labels = batch
            if self.binirize_labels:
                inputs = {"target": (labels > 0).float()}
            else:
                inputs = {"target": labels}

        output = {"clipwise_logits": model(wave)}

        loss_v = self.loss_f(output["clipwise_logits"], labels)

        if self.use_slected_indices:
            loss_v = loss_v[:, self.selected_indices]

        if self.use_weights:
            loss_v = (self.weights * loss_v).sum(dim=1).mean()
        elif self.class_sum:
            loss_v = loss_v.sum(dim=1).mean()

        losses = {"loss": loss_v}

        return losses, inputs, output


class NoCallDetectForwardLongShort(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.loss_f = nn.BCEWithLogitsLoss()

    def forward(self, runner, batch):

        mode = "train" if runner.loader_key == "train" else "val"
        model = get_mode_model(runner.model, mode=mode)

        if runner.loader_key == "infer":
            wave, labels, _, _, _ = batch
            inputs = {"target": labels}
        else:
            wave, labels = batch
            inputs = {"target": labels}

        output = model(wave)

        loss_v = self.loss_f(output["clipwise_logits_long"], labels)

        losses = {"loss": loss_v}

        return losses, inputs, output
