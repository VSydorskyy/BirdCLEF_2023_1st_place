import torch
from catalyst.core.callback import Callback, CallbackOrder


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    buf1 = dict(model1.named_buffers())
    buf2 = dict(model2.named_buffers())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

    for k in buf1.keys():
        # some buffers contain integers, which refer to number of processed
        # samples or batches. These values we have to copy
        if torch.is_floating_point(buf1[k]):
            buf1[k].data.mul_(decay).add_(buf2[k].data, alpha=1 - decay)
        else:
            buf1[k].data = buf2[k].data


class ModelEMA(Callback):
    def __init__(self, ema_coef=0.999, model_mapping=None):
        super().__init__(order=CallbackOrder.external)
        self.ema_coef = ema_coef
        self.model_mapping = model_mapping

    def on_batch_end(self, runner):
        if runner.loader_key == "train":
            if self.model_mapping is not None:
                for (
                    val_model_name,
                    train_model_name,
                ) in self.model_mapping.items():
                    accumulate(
                        runner.model[val_model_name],
                        runner.model[train_model_name],
                        self.ema_coef,
                    )
            else:
                accumulate(
                    runner.model["val"],
                    runner.model["train"],
                    self.ema_coef,
                )
