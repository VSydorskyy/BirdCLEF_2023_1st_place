import torch
import torch.nn as nn


class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction="none")(preds, targets)
        probas = torch.sigmoid(preds)
        loss = (
            targets * self.alpha * (1.0 - probas) ** self.gamma * bce_loss
            + (1.0 - targets) * probas**self.gamma * bce_loss
        )
        loss = loss.mean()
        return loss


class BCEFocal2WayLoss(nn.Module):
    def __init__(
        self,
        weights=[1, 1],
        clipwise_name="clipwise_logits_long",
        framewise_name="framewise_logits_long",
    ):
        super().__init__()
        self.focal = BCEFocalLoss()
        self.clipwise_name = clipwise_name
        self.framewise_name = framewise_name
        self.weights = weights

    def forward(self, input, target):
        input_ = input[self.clipwise_name]

        framewise_output = input[self.framewise_name]
        clipwise_output_with_max, _ = framewise_output.max(dim=1)

        loss = self.focal(input_, target)
        aux_loss = self.focal(clipwise_output_with_max, target)

        return self.weights[0] * loss + self.weights[1] * aux_loss
