import numpy as np
import torch.nn as nn
from torchaudio.transforms import FrequencyMasking, TimeMasking


class CustomMasking(nn.Module):
    def __init__(
        self, mask_max_length: int, mask_max_masks: int, p=1.0, inplace=True
    ):
        super().__init__()
        assert isinstance(mask_max_masks, int) and mask_max_masks > 0
        self.mask_max_masks = mask_max_masks
        self.mask_max_length = mask_max_length
        self.mask_module = None
        self.p = p
        self.inplace = inplace

    def forward(self, x):
        if not self.inplace:
            output = x.clone()
        for i in range(x.shape[0]):
            if np.random.binomial(n=1, p=self.p):
                n_applies = np.random.randint(
                    low=1, high=self.mask_max_masks + 1
                )
                for _ in range(n_applies):
                    if self.inplace:
                        x[i : i + 1] = self.mask_module(x[i : i + 1])
                    else:
                        output[i : i + 1] = self.mask_module(output[i : i + 1])
        if self.inplace:
            return x
        else:
            return output


class CustomTimeMasking(CustomMasking):
    def __init__(
        self, mask_max_length: int, mask_max_masks: int, p=1.0, inplace=True
    ):
        super().__init__(
            mask_max_length=mask_max_length,
            mask_max_masks=mask_max_masks,
            p=p,
            inplace=inplace,
        )
        self.mask_module = TimeMasking(time_mask_param=mask_max_length)


class CustomFreqMasking(CustomMasking):
    def __init__(
        self, mask_max_length: int, mask_max_masks: int, p=1.0, inplace=True
    ):
        super().__init__(
            mask_max_length=mask_max_length,
            mask_max_masks=mask_max_masks,
            p=p,
            inplace=inplace,
        )
        self.mask_module = FrequencyMasking(freq_mask_param=mask_max_length)
