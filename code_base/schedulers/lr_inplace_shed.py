import warnings

from torch.optim.lr_scheduler import LambdaLR


class LambdaLRInplace(LambdaLR):
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`."
            )

        return [
            lmbda(self.last_epoch)
            for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)
        ]
