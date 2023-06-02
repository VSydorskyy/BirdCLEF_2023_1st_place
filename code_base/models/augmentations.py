import torch
import torch.nn as nn


class RandomFiltering(nn.Module):
    def __init__(
        self,
        n_bands: int = 4,
        min_db: float = -15,
        is_wave: bool = True,
        normalize_wave: bool = False,
        n_fft: int = 1024,
        hop_length: int = 512,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.n_bands = n_bands
        self.min_db = min_db
        self.is_wave = is_wave
        self.normalize_wave = normalize_wave
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.eps = eps
        self.register_buffer("hann_window", torch.hann_window(self.n_fft))

    def spec_forward(self, s):
        # s shape : B, Bins, Time
        filter_points = (
            torch.rand(s.shape[0], self.n_bands, device=s.device) * self.min_db
        )
        # Interpolate requires channels:
        # mini-batch x channels x [optional depth] x [optional height] x width
        filter_coeffs = nn.functional.interpolate(
            filter_points.unsqueeze(1), mode="linear", size=s.shape[1]
        ).squeeze(1)
        # DB to Amplitude
        filter_coeffs = 10 ** (filter_coeffs / 20)
        return s * filter_coeffs.unsqueeze(-1)

    def forward(self, x):
        if self.is_wave:
            complex_spec = torch.stft(
                x,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=None,
                window=self.hann_window,
                center=True,
                pad_mode="reflect",
                normalized=False,
                onesided=None,
                return_complex=True,
            )
            filtered_spec = self.spec_forward(complex_spec)
            filtered_wave = torch.istft(
                filtered_spec,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=None,
                window=self.hann_window,
                center=True,
                normalized=False,
                onesided=None,
                length=None,
                return_complex=False,
            )
            if self.normalize_wave:
                filtered_wave /= (
                    filtered_wave.abs().max(axis=1, keepdims=True)[0]
                    + self.eps
                )
            return filtered_wave
        else:
            return self.spec_forward(x)
