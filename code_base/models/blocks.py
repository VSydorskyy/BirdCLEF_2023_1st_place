import math
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from nnAudio.features.stft import STFT as nnAudioSTFT
    from nnAudio.features.stft import STFTBase
    from nnAudio.utils import create_fourier_kernels
except:
    print("`nnAudioSTFT` was not imported")
from typing import Optional

from torchaudio.transforms import MelScale

EPSILON_FP16 = 1e-5


class NormalizeMelSpec(nn.Module):
    def __init__(self, eps=1e-6, exportable=False):
        super().__init__()
        self.eps = eps
        self.exportable = exportable

    def forward(self, X):
        mean = X.mean((1, 2), keepdim=True)
        std = X.std((1, 2), keepdim=True)
        Xstd = (X - mean) / (std + self.eps)
        if self.exportable:
            norm_max = torch.amax(Xstd, dim=(1, 2), keepdim=True)
            norm_min = torch.amin(Xstd, dim=(1, 2), keepdim=True)
            return (Xstd - norm_min) / (norm_max - norm_min + self.eps)
        else:
            norm_min, norm_max = (
                Xstd.min(-1)[0].min(-1)[0],
                Xstd.max(-1)[0].max(-1)[0],
            )
            fix_ind = (norm_max - norm_min) > self.eps * torch.ones_like(
                (norm_max - norm_min)
            )
            V = torch.zeros_like(Xstd)
            if fix_ind.sum():
                V_fix = Xstd[fix_ind]
                norm_max_fix = norm_max[fix_ind, None, None]
                norm_min_fix = norm_min[fix_ind, None, None]
                V_fix = torch.max(
                    torch.min(V_fix, norm_max_fix),
                    norm_min_fix,
                )
                V_fix = (V_fix - norm_min_fix) / (norm_max_fix - norm_min_fix)
                V[fix_ind] = V_fix
            return V


class Swish(nn.Module):
    def forward(self, x):
        gate = torch.sigmoid(x)
        gate = torch.clamp(gate, min=EPSILON_FP16, max=1.0 - EPSILON_FP16)
        return x * gate


def gem_freq(x, p=3, eps=1e-6, exportable=False):
    if exportable:
        return x.clamp(min=eps).pow(p).mean(2, keepdims=True).pow(1.0 / p)
    else:
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), 1)).pow(
            1.0 / p
        )


class GeMFreq(nn.Module):
    def __init__(self, p=3, eps=1e-6, exportable=False):
        super().__init__()
        self.p = torch.nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self.exportable = exportable

    def forward(self, x):
        return gem_freq(x, p=self.p, eps=self.eps, exportable=self.exportable)


class GeMFreqFixed(nn.Module):
    def __init__(self, time_kernel_size, p=3, eps=1e-6):
        super().__init__()
        self.p = torch.nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self.avg_pool = nn.AvgPool2d(kernel_size=(1, time_kernel_size))

    def forward(self, x):
        return self.avg_pool(
            x.clamp(min=self.eps).pow(self.p).mean(2, keepdims=True)
        ).pow(1.0 / self.p)


class Clasifier(nn.Module):
    def __init__(
        self,
        classifier_type,
        nn_embed_size=None,
        classes_num=None,
        second_dropout_rate=None,
        hidden_dims=None,
        first_dropout_rate=None,
    ):
        super().__init__()

        if classifier_type == "relu":
            self.classifier = nn.Sequential(
                nn.Linear(nn_embed_size, hidden_dims),
                nn.ReLU(),
                nn.Dropout(p=first_dropout_rate),
                nn.Linear(hidden_dims, hidden_dims),
                nn.ReLU(),
                nn.Dropout(p=second_dropout_rate),
                nn.Linear(hidden_dims, classes_num),
            )
        elif classifier_type == "elu":
            self.classifier = nn.Sequential(
                nn.Dropout(first_dropout_rate),
                nn.Linear(nn_embed_size, hidden_dims),
                nn.ELU(),
                nn.Dropout(second_dropout_rate),
                nn.Linear(hidden_dims, classes_num),
            )
        elif classifier_type == "swish":
            self.classifier = nn.Sequential(
                nn.Dropout(first_dropout_rate),
                nn.Linear(nn_embed_size, hidden_dims),
                Swish(),
                nn.Dropout(second_dropout_rate),
                nn.Linear(hidden_dims, classes_num),
            )
        elif classifier_type == "dima":
            self.classifier = nn.Sequential(
                nn.BatchNorm1d(nn_embed_size),
                nn.Linear(nn_embed_size, hidden_dims),
                nn.BatchNorm1d(hidden_dims),
                nn.PReLU(hidden_dims),
                nn.Dropout(p=second_dropout_rate),
                nn.Linear(hidden_dims, classes_num),
            )
        elif classifier_type == "prelu":
            self.classifier = nn.Sequential(
                nn.Dropout(first_dropout_rate),
                nn.Linear(nn_embed_size, hidden_dims),
                nn.PReLU(hidden_dims),
                nn.Dropout(p=second_dropout_rate),
                nn.Linear(hidden_dims, classes_num),
            )
        elif classifier_type == "drop_linear":
            self.classifier = nn.Sequential(
                nn.Dropout(p=second_dropout_rate),
                nn.Linear(nn_embed_size, classes_num),
            )
        elif classifier_type == "identity":
            self.classifier = nn.Identity()
        else:
            raise ValueError("Invalid classifier_type")

    def forward(self, input):
        return self.classifier(input)


class PoolingLayer(nn.Module):
    def __init__(self, pool_type: str, p=3, eps=1e-6):
        super().__init__()

        self.pool_type = pool_type

        if self.pool_type == "AdaptiveAvgPool2d":
            self.pool_layer = nn.AdaptiveAvgPool2d((1, 1))
        elif self.pool_type == "GeM":
            self.pool_layer = nn.AdaptiveAvgPool2d((1, 1))
            self.p = torch.nn.Parameter(torch.ones(1) * p)
            self.eps = eps
        else:
            raise RuntimeError(f"{self.pool_type} is invalid pool_type")

    def forward(self, x):
        bs, ch, h, w = x.shape
        if self.pool_type == "AdaptiveAvgPool2d":
            x = self.pool_layer(x)
            x = x.view(bs, ch)
        elif self.pool_type == "GeM":
            x = self.pool_layer(x.clamp(min=self.eps).pow(self.p)).pow(
                1.0 / self.p
            )
            x = x.view(bs, ch)
        return x


class AttHead(nn.Module):
    def __init__(
        self,
        in_chans,
        hidden_chans=512,
        p=0.5,
        num_class=397,
        train_period=15.0,
        infer_period=5.0,
        infer_framewise_max_coef=0.5,
        output_type=None,
        exportable=False,
        rnn_config=None,
    ):
        super().__init__()
        self.train_period = train_period
        self.infer_period = infer_period
        self.infer_framewise_max_coef = infer_framewise_max_coef
        self.output_type = output_type
        if isinstance(in_chans, list):
            self.pooling = nn.ModuleList(
                [
                    GeMFreqFixed(time_kernel_size=k * 2 if k > 0 else 1)
                    for k in list(reversed(range(len(in_chans))))
                ]
            )
            self.deep_supervision = True
            in_chans = sum(in_chans)
        else:
            self.pooling = GeMFreq(exportable=exportable)
            self.deep_supervision = False

        if rnn_config is not None:
            self.rnn = nn.GRU(
                input_size=in_chans,
                hidden_size=rnn_config["hidden_size"],
                num_layers=rnn_config["num_layers"],
                batch_first=True,
                bidirectional=rnn_config["bidirectional"],
            )
            if rnn_config["bidirectional"]:
                in_chans = rnn_config["hidden_size"] * 2
            else:
                in_chans = rnn_config["hidden_size"]
        else:
            self.rnn = None
        self.dense_layers = nn.Sequential(
            nn.Dropout(p / 2),
            nn.Linear(in_chans, hidden_chans),
            nn.ReLU(),
            nn.Dropout(p),
        )
        self.attention = nn.Conv1d(
            in_channels=hidden_chans,
            out_channels=num_class,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.fix_scale = nn.Conv1d(
            in_channels=hidden_chans,
            out_channels=num_class,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(self, feat):
        if self.deep_supervision:
            feat = torch.cat(
                [pool(one_feat) for pool, one_feat in zip(self.pooling, feat)],
                dim=1,
            )
        else:
            feat = self.pooling(feat)
        if self.rnn is not None:
            feat = feat.squeeze(-2).permute(0, 2, 1)
            feat, _ = self.rnn(feat)
        else:
            feat = feat.squeeze(-2).permute(0, 2, 1)  # (bs, time, ch)
        feat = self.dense_layers(feat).permute(0, 2, 1)  # (bs, 512, time)

        time_att = torch.tanh(self.attention(feat))
        feat_v = self.fix_scale(feat)

        assert self.train_period >= self.infer_period

        if self.training or self.train_period == self.infer_period:
            if self.output_type == "timewise_pred_max":
                return torch.max(torch.sigmoid(feat_v), dim=-1)[0]
            clipwise_pred_long = torch.sum(
                torch.sigmoid(feat_v) * torch.softmax(time_att, dim=-1),
                dim=-1,
            )
            if self.output_type == "clipwise_pred_long":
                return clipwise_pred_long
            if self.output_type == "clipwise_timewisemax_pred_short":
                return torch.max(torch.sigmoid(feat_v), dim=-1)[
                    0
                ] * self.infer_framewise_max_coef + clipwise_pred_long * (
                    1 - self.infer_framewise_max_coef
                )
            clipwise_logits_long = torch.sum(
                feat_v * torch.softmax(time_att, dim=-1),
                dim=-1,
            )
            if self.output_type == "clipwise_logits_long":
                return clipwise_logits_long
            return {
                # Clipwise
                "clipwise_pred_long": clipwise_pred_long,
                "clipwise_logits_long": clipwise_logits_long,
                # # Short
                # "clipwise_pred_short": torch.scalar_tensor(
                #     0, device=clipwise_logits_long.device
                # ),
                # "clipwise_logits_short": torch.scalar_tensor(
                #     0, device=clipwise_logits_long.device
                # ),
                # Framewise
                "framewise_logits_long": feat_v.permute(0, 2, 1),
            }
        else:
            # Compute all frame pred
            framewise_pred_long = torch.sigmoid(feat_v)
            clipwise_logits_long = torch.sum(
                feat_v * torch.softmax(time_att, dim=-1), dim=-1
            )
            if self.output_type == "clipwise_logits_long":
                return clipwise_logits_long
            clipwise_pred_long = torch.sum(
                framewise_pred_long * torch.softmax(time_att, dim=-1), dim=-1
            )
            if self.output_type == "clipwise_pred_long":
                return clipwise_pred_long
            # Compute start and end of small frame
            feat_time = feat.size(-1)
            start = (
                feat_time / 2
                - feat_time * (self.infer_period / self.train_period) / 2
            )
            end = start + feat_time * (self.infer_period / self.train_period)
            start = int(start)
            end = int(end)
            # Cut attention and values
            feat_v_short = feat_v[:, :, start:end]
            time_att_short = time_att[:, :, start:end]
            # Compute small frame pred
            framewise_pred_short = torch.sigmoid(feat_v_short)
            clipwise_logits_short = torch.sum(
                feat_v_short * torch.softmax(time_att_short, dim=-1), dim=-1
            )
            if self.output_type == "clipwise_logits_short":
                return clipwise_logits_short
            clipwise_pred_short = torch.sum(
                framewise_pred_short * torch.softmax(time_att_short, dim=-1),
                dim=-1,
            )
            if self.output_type == "clipwise_pred_short":
                return clipwise_pred_short
            framewise_clipwise_pred_short = framewise_pred_short.max(axis=2)[
                0
            ] * self.infer_framewise_max_coef + clipwise_pred_short * (
                1 - self.infer_framewise_max_coef
            )
            return {
                "clipwise_pred_long": clipwise_pred_long,
                "clipwise_logits_long": clipwise_logits_long,
                "clipwise_pred_short": clipwise_pred_short,
                "clipwise_logits_short": clipwise_logits_short,
                # Framewise
                "framewise_logits_long": feat_v.permute(0, 2, 1),
                "framewise_logits_short": feat_v_short.permute(0, 2, 1),
                "framewise_pred_short": framewise_pred_short.permute(0, 2, 1),
                "framewise_atten_short": time_att_short.permute(0, 2, 1),
                # Combined
                "framewise_clipwise_pred_short": framewise_clipwise_pred_short,
            }


class TraceableMelspec(nn.Module):
    def __init__(
        self,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        power: float = 2.0,
        normalized: bool = False,
        center: bool = True,
        pad_mode: str = "reflect",
        # Mel params
        n_mels: int = 128,
        sample_rate: int = 16000,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        n_fft: int = 400,
        norm: Optional[str] = None,
        mel_scale: str = "htk",
        # Add params
        trainable: bool = False,
        quantizable: bool = False,
    ):
        super().__init__()
        if quantizable:
            self.spectrogram = QuantizableSTFT(
                n_fft=n_fft,
                win_length=win_length,
                freq_bins=None,
                hop_length=hop_length,
                window="hann",
                freq_scale="no",
                # Do not define `fmin` and `fmax`, because freq_scale = "no"
                center=center,
                pad_mode=pad_mode,
                sr=sample_rate,
                trainable=trainable,
                output_format="Complex",
                verbose=True,
            )
        else:
            self.spectrogram = nnAudioSTFT(
                n_fft=n_fft,
                win_length=win_length,
                freq_bins=None,
                hop_length=hop_length,
                window="hann",
                freq_scale="no",
                # Do not define `fmin` and `fmax`, because freq_scale = "no"
                center=center,
                pad_mode=pad_mode,
                iSTFT=False,
                sr=sample_rate,
                trainable=trainable,
                output_format="Complex",
                verbose=True,
            )
        self.normalized = normalized
        self.power = power
        self.register_buffer(
            "window",
            torch.hann_window(win_length if win_length is not None else n_fft),
        )
        self.trainable = trainable
        self.mel_scale = MelScale(
            n_mels, sample_rate, f_min, f_max, n_fft // 2 + 1, norm, mel_scale
        )

    def forward(self, x):
        spec_f = self.spectrogram(x)
        if self.normalized:
            spec_f /= self.window.pow(2.0).sum().sqrt()
        if self.power is not None:
            # prevent Nan gradient when sqrt(0) due to output=0
            # Taken from nnAudio.features.stft.STFT
            eps = 1e-8 if self.trainable else 0.0
            spec_f = torch.sqrt(
                spec_f[:, :, :, 0].pow(2) + spec_f[:, :, :, 1].pow(2) + eps
            )
            if self.power != 1.0:
                spec_f = spec_f.pow(self.power)
        mel_spec = self.mel_scale(spec_f)
        return mel_spec


class QuantizableSTFT(STFTBase):
    def __init__(
        self,
        n_fft=2048,
        win_length=None,
        freq_bins=None,
        hop_length=None,
        window="hann",
        freq_scale="no",
        center=True,
        pad_mode="reflect",
        fmin=50,
        fmax=6000,
        sr=22050,
        trainable=False,
        output_format="Complex",
        verbose=True,
    ):

        super().__init__()

        # Trying to make the default setting same as librosa
        if win_length == None:
            win_length = n_fft
        if hop_length == None:
            hop_length = int(win_length // 4)

        self.output_format = output_format
        self.trainable = trainable
        self.stride = hop_length
        self.center = center
        self.pad_mode = pad_mode
        self.n_fft = n_fft
        self.freq_bins = freq_bins
        self.trainable = trainable
        self.pad_amount = self.n_fft // 2
        self.window = window
        self.win_length = win_length
        self.trainable = trainable
        start = time()

        # Create filter windows for stft
        (
            kernel_sin,
            kernel_cos,
            self.bins2freq,
            self.bin_list,
            window_mask,
        ) = create_fourier_kernels(
            n_fft,
            win_length=win_length,
            freq_bins=freq_bins,
            window=window,
            freq_scale=freq_scale,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            verbose=verbose,
        )

        kernel_sin = torch.tensor(kernel_sin, dtype=torch.float)
        kernel_cos = torch.tensor(kernel_cos, dtype=torch.float)

        # Applying window functions to the Fourier kernels
        window_mask = torch.tensor(window_mask)
        wsin = kernel_sin * window_mask
        wcos = kernel_cos * window_mask

        self.spec_imag_conv = nn.Conv1d(
            wsin.shape[1],
            wsin.shape[0],
            wsin.shape[2],
            stride=self.stride,
            bias=False,
        )
        self.spec_real_conv = nn.Conv1d(
            wcos.shape[1],
            wcos.shape[0],
            wcos.shape[2],
            stride=self.stride,
            bias=False,
        )
        # Set the weights and bias manually
        with torch.no_grad():
            self.spec_imag_conv.weight.copy_(wsin)
            self.spec_real_conv.weight.copy_(wcos)

        if self.trainable == False:
            for param in self.spec_imag_conv.parameters():
                param.requires_grad = False
            for param in self.spec_real_conv.parameters():
                param.requires_grad = False

        if verbose == True:
            print(
                "STFT kernels created, time used = {:.4f} seconds".format(
                    time() - start
                )
            )
        else:
            pass

        if self.center:
            if self.pad_mode == "constant":
                self.padding_node = nn.ConstantPad1d(self.pad_amount, 0)
            elif self.pad_mode == "reflect":
                self.padding_node = nn.ReflectionPad1d(self.pad_amount)

    def forward(self, x):
        """
        Convert a batch of waveforms to spectrograms.

        Parameters
        ----------
        x : torch tensor
            Input signal should be in either of the following shapes.\n
            1. ``(len_audio)``\n
            2. ``(num_audio, len_audio)``\n
            3. ``(num_audio, 1, len_audio)``
            It will be automatically broadcast to the right shape
        """
        self.num_samples = x.shape[-1]

        x = x[:, None, :]
        if self.center:
            x = self.padding_node(x)
        spec_imag = self.spec_imag_conv(x)
        spec_real = self.spec_real_conv(x)

        # remove redundant parts
        spec_real = spec_real[:, : self.freq_bins, :]
        spec_imag = spec_imag[:, : self.freq_bins, :]

        if self.output_format == "Magnitude":
            spec = spec_real.pow(2) + spec_imag.pow(2)
            if self.trainable == True:
                return torch.sqrt(
                    spec + 1e-8
                )  # prevent Nan gradient when sqrt(0) due to output=0
            else:
                return torch.sqrt(spec)

        elif self.output_format == "Complex":
            return torch.stack(
                (spec_real, -spec_imag), -1
            )  # Remember the minus sign for imaginary part

        elif self.output_format == "Phase":
            return torch.atan2(
                -spec_imag + 0.0, spec_real
            )  # +0.0 removes -0.0 elements, which leads to error in calculating phase


class QuantizableAmplitudeToDB(torch.nn.Module):
    r"""Turn a tensor from the power/amplitude scale to the decibel scale.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    This output depends on the maximum value in the input tensor, and so
    may return different values for an audio clip split into snippets vs. a
    a full clip.

    Args:
        stype (str, optional): scale of input tensor (``"power"`` or ``"magnitude"``). The
            power being the elementwise square of the magnitude. (Default: ``"power"``)
        top_db (float or None, optional): minimum negative cut-off in decibels.  A reasonable
            number is 80. (Default: ``None``)

    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav", normalize=True)
        >>> transform = transforms.AmplitudeToDB(stype="amplitude", top_db=80)
        >>> waveform_db = transform(waveform)
    """
    __constants__ = ["multiplier", "amin", "ref_value", "db_multiplier"]

    def __init__(
        self, stype: str = "power", top_db: Optional[float] = None
    ) -> None:
        super().__init__()
        self.stype = stype
        if top_db is not None and top_db < 0:
            raise ValueError("top_db must be positive value")
        self.top_db = top_db
        self.multiplier = 10.0 if stype == "power" else 20.0
        self.amin = 1e-10
        self.ref_value = 1.0
        self.db_multiplier = math.log10(max(self.amin, self.ref_value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Numerically stable implementation from Librosa.

        https://librosa.org/doc/latest/generated/librosa.amplitude_to_db.html

        Args:
            x (Tensor): Input tensor before being converted to decibel scale.

        Returns:
            Tensor: Output tensor in decibel scale.
        """
        x_db = self.multiplier * torch.log10(torch.clamp(x, min=self.amin))
        x_db -= self.multiplier * self.db_multiplier

        if self.top_db is not None:
            # Expand batch
            shape = x_db.size()
            x_db = x_db.reshape(-1, shape[-3], shape[-2], shape[-1])

            x_db = torch.max(
                x_db,
                (x_db.amax(dim=(-3, -2, -1)) - self.top_db).view(-1, 1, 1, 1),
            )

            # Repack batch
            x_db = x_db.reshape(shape)

        return x_db
