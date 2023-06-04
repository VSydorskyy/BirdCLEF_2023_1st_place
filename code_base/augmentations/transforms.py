import math
from glob import glob
from os.path import join as pjoin
from warnings import warn

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import colorednoise as cn
except:
    print("colorednoise package is missing")

from ..utils import parallel_librosa_load


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray):
        for trns in self.transforms:
            y = trns(y)
        return y


class OneOf:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray):
        n_trns = len(self.transforms)
        trns_idx = np.random.choice(n_trns)
        trns = self.transforms[trns_idx]
        y = trns(y)
        return y


class AudioTransform:
    def __init__(self, always_apply=False, p=0.5):
        self.always_apply = always_apply
        self.p = p

    def __call__(self, y: np.ndarray):
        if self.always_apply:
            return self.apply(y)
        else:
            if np.random.rand() < self.p:
                return self.apply(y)
            else:
                return y

    def apply(self, y: np.ndarray):
        raise NotImplementedError


class NoiseInjection(AudioTransform):
    def __init__(
        self,
        always_apply=False,
        p=0.5,
        max_noise_level=0.5,
        sr=32000,
        normalize=True,
    ):
        super().__init__(always_apply, p)

        self.noise_level = (0.0, max_noise_level)
        self.sr = sr
        self.normalize = normalize

    def apply(self, y: np.ndarray, **params):
        noise_level = np.random.uniform(*self.noise_level)
        noise = np.random.randn(len(y))
        augmented = (y + noise * noise_level).astype(y.dtype)
        if self.normalize:
            augmented = librosa.util.normalize(augmented)
        return augmented


class GaussianNoise(AudioTransform):
    def __init__(
        self,
        always_apply=False,
        p=0.5,
        min_snr=5,
        max_snr=20,
        sr=32000,
        normalize=True,
        verbose=False,
    ):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr
        self.sr = sr
        self.normalize = normalize
        self.verbose = verbose

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        if self.verbose:
            print(f"GaussianNoise. SNR = {snr}")
        a_signal = np.sqrt(y**2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        white_noise = np.random.randn(len(y))
        a_white = np.sqrt(white_noise**2).max()
        augmented = (y + white_noise * 1 / a_white * a_noise).astype(y.dtype)
        if self.normalize:
            augmented = librosa.util.normalize(augmented)
        return augmented


class PinkNoise(AudioTransform):
    def __init__(
        self,
        always_apply=False,
        p=0.5,
        min_snr=5,
        max_snr=20,
        sr=32000,
        normalize=True,
        verbose=False,
    ):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr
        self.sr = sr
        self.normalize = normalize
        self.verbose = verbose

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        if self.verbose:
            print(f"PinkNoise. SNR = {snr}")
        a_signal = np.sqrt(y**2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        pink_noise = cn.powerlaw_psd_gaussian(1, len(y))
        a_pink = np.sqrt(pink_noise**2).max()
        augmented = (y + pink_noise * 1 / a_pink * a_noise).astype(y.dtype)
        if self.normalize:
            augmented = librosa.util.normalize(augmented)
        return augmented


class PitchShift(AudioTransform):
    def __init__(
        self, always_apply=False, p=0.5, max_range=5, sr=32000, normalize=True
    ):
        super().__init__(always_apply, p)
        self.max_range = max_range
        self.sr = sr
        self.normalize = normalize

    def apply(self, y: np.ndarray, **params):
        n_steps = np.random.randint(-self.max_range, self.max_range)
        augmented = librosa.effects.pitch_shift(y, self.sr, n_steps)
        if self.normalize:
            augmented = librosa.util.normalize(augmented)
        return augmented


class TimeStretch(AudioTransform):
    def __init__(
        self, always_apply=False, p=0.5, max_rate=1, sr=32000, normalize=True
    ):
        super().__init__(always_apply, p)
        self.max_rate = max_rate
        self.sr = sr
        self.normalize = normalize

    def apply(self, y: np.ndarray, **params):
        rate = np.random.uniform(0, self.max_rate)
        augmented = librosa.effects.time_stretch(y, rate)
        if self.normalize:
            augmented = librosa.util.normalize(augmented)
        return augmented


def _db2float(db: float, amplitude=True):
    if amplitude:
        return 10 ** (db / 20)
    else:
        return 10 ** (db / 10)


def volume_down(y: np.ndarray, db: float):
    """
    Low level API for decreasing the volume
    Parameters
    ----------
    y: numpy.ndarray
        stereo / monaural input audio
    db: float
        how much decibel to decrease
    Returns
    -------
    applied: numpy.ndarray
        audio with decreased volume
    """
    applied = y * _db2float(-db)
    return applied


def volume_up(y: np.ndarray, db: float):
    """
    Low level API for increasing the volume
    Parameters
    ----------
    y: numpy.ndarray
        stereo / monaural input audio
    db: float
        how much decibel to increase
    Returns
    -------
    applied: numpy.ndarray
        audio with increased volume
    """
    applied = y * _db2float(db)
    return applied


class RandomVolume(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, limit=10, normalize=True):
        super().__init__(always_apply, p)
        self.limit = limit
        self.normalize = normalize

    def apply(self, y: np.ndarray, **params):
        db = np.random.uniform(-self.limit, self.limit)
        if db >= 0:
            augmented = volume_up(y, db)
        else:
            augmented = volume_down(y, db)
        if self.normalize:
            augmented = librosa.util.normalize(augmented)
        return augmented


class CosineVolume(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, limit=10, normalize=True):
        super().__init__(always_apply, p)
        self.limit = limit
        self.normalize = normalize

    def apply(self, y: np.ndarray, **params):
        db = np.random.uniform(-self.limit, self.limit)
        cosine = np.cos(np.arange(len(y)) / len(y) * np.pi * 2)
        dbs = _db2float(cosine * db)
        augmented = y * dbs
        if self.normalize:
            augmented = librosa.util.normalize(augmented)
        return augmented


class BackgroundNoise(AudioTransform):
    def __init__(
        self,
        background_regex=None,
        esc50_root=None,
        esc50_df_path=None,
        esc50_cats_to_include=None,
        always_apply=False,
        p=0.5,
        min_level=0.25,
        max_level=0.75,
        sr=32000,
        normalize=True,
        verbose=False,
        glob_recursive=False,
    ):
        super().__init__(always_apply, p)
        assert min_level < max_level
        assert 0 < min_level < 1
        assert 0 < max_level < 1
        if background_regex is None and (
            esc50_root is None or esc50_df_path is None
        ):
            raise ValueError(
                "background_regex OR esc50_root AND esc50_df_path should be defined"
            )
        if background_regex is not None:
            sample_names = glob(background_regex, recursive=glob_recursive)
        else:
            sample_df = pd.read_csv(esc50_df_path)
            if esc50_cats_to_include is not None:
                sample_df = sample_df[
                    sample_df.category.isin(esc50_cats_to_include)
                ]
            sample_names = [
                pjoin(esc50_root, el) for el in sample_df.filename.tolist()
            ]
        self.samples = parallel_librosa_load(
            sample_names,
            return_sr=False,
            sr=sr,
            do_normalize=True,
        )
        self.normalize = normalize
        self.min_max_levels = (min_level, max_level)
        self.verbose = verbose

    @staticmethod
    def crop_sample(sample, crop_shape):
        start = np.random.randint(0, sample.shape[0] - crop_shape)
        return sample[start : start + crop_shape]

    def apply(self, y: np.ndarray, **params):
        # TODO: It is a dirty hack BUT by some reasons we have corrupted samples
        # In order not to use it make sure that all samples are Okey
        back_sample = None
        while back_sample is None:
            back_sample = self.samples[np.random.randint(len(self.samples))]

        if y.shape[0] < back_sample.shape[0]:
            back_sample = self.crop_sample(back_sample, y.shape[0])
        elif y.shape[0] > back_sample.shape[0]:
            repeat_times = math.ceil(y.shape[0] / back_sample.shape[0])
            back_sample = np.concatenate([back_sample] * repeat_times)
            if back_sample.shape[0] > y.shape[0]:
                back_sample = self.crop_sample(back_sample, y.shape[0])

        back_amp = np.random.uniform(*self.min_max_levels)
        if self.verbose:
            print(f"BackgroundNoise. back_amp: {back_amp}")
        augmented = y * (1 - back_amp) + back_sample * back_amp

        if self.normalize:
            augmented = librosa.util.normalize(augmented)
        return augmented
