import warnings
from glob import glob
from os.path import join as pjoin

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..utils.constants import SAMPLE_RATE


class AddBackgoundFromSoundScapes:
    def __init__(
        self,
        audio_folder_soundscape,
        df_path_soundscape,
        p=0.5,
        always_apply=False,
        audio_ext=".ogg",
        amplitude=None,
        use_only_long_samples=True,
    ):
        audio_pathes = glob(pjoin(audio_folder_soundscape, "*" + audio_ext))
        self.audio_files = {
            el: librosa.util.normalize(librosa.load(el, sr=None)[0])
            for el in tqdm(audio_pathes)
        }
        self.soundscape_df = pd.read_csv(df_path_soundscape)
        self.soundscape_df["s_e"] = self.soundscape_df["s_e"].apply(eval)
        self.soundscape_df["s_e"] = self.soundscape_df["s_e"].apply(
            lambda x: [x[0] * SAMPLE_RATE, x[1] * SAMPLE_RATE]
        )
        self.soundscape_df["len"] = self.soundscape_df["s_e"].apply(
            lambda x: (x[1] - x[0])
        )

        self.p = p
        self.always_apply = always_apply
        self.use_only_long_samples = use_only_long_samples
        if amplitude is None:
            pass
        elif isinstance(amplitude, float):
            if amplitude <= 0.0 or amplitude >= 1.0:
                raise RuntimeError(
                    "amplitude should be normalized to (0,1). "
                    "0 and 1 are also invalid values"
                )
        elif isinstance(amplitude, tuple):
            pass
        else:
            raise RuntimeError("invalid amplitude")
        self.amplitude = amplitude

    def _trim_background(self, input, sample_len):
        if input.shape[0] < sample_len:
            pad_len = sample_len - input.shape[0]
            return np.pad(input, ((pad_len, 0)))
        elif input.shape[0] > sample_len:
            start = np.random.randint(0, input.shape[0] - sample_len)
            return input[start : start + sample_len]
        else:
            return input

    def __call__(self, input, target):

        if self.always_apply or np.random.binomial(n=1, p=self.p):

            if self.use_only_long_samples:
                sub_df = self.soundscape_df[
                    self.soundscape_df["len"] >= len(input)
                ].reset_index(drop=True)
            else:
                sub_df = self.soundscape_df

            if len(sub_df) == 0:
                warnings.warn(
                    "All samples of soundscape_df do not much length of the input"
                )
                return input, target

            selected_row = sub_df.iloc[np.random.randint(0, sub_df.shape[0])]

            background_au = self.audio_files[selected_row["path"]]
            background_au = background_au[
                selected_row["s_e"][0] : selected_row["s_e"][1]
            ]
            background_au = librosa.util.normalize(
                self._trim_background(background_au, len(input))
            )

            if self.amplitude is None:
                input = librosa.util.normalize(input + background_au)
            elif isinstance(self.amplitude, float):
                input = librosa.util.normalize(
                    input * (1 - self.amplitude)
                    + background_au * self.amplitude
                )
            elif isinstance(self.amplitude, tuple):
                amplitude = np.random.uniform(
                    low=self.amplitude[0], high=self.amplitude[1]
                )
                input = librosa.util.normalize(
                    input * (1 - amplitude) + background_au * amplitude
                )

        return input, target
