import math
import warnings
from os.path import join as pjoin
from os.path import relpath, splitext
from time import time

import h5py
import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ..utils import load_json, load_pp_audio, parallel_librosa_load

DEFAULT_TARGET = 0
EPS = 1e-5
MAX_RATING = 5.0


class WaveDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        label_str2int_mapping_path,
        replace_pathes=None,
        df=None,
        add_df_paths=None,
        add_root=None,
        target_col="primary_label",
        sec_target_col="secondary_labels",
        name_col="filename",
        rating_col=None,
        sample_rate=32_000,
        segment_len=5.0,
        precompute=False,
        early_aug=None,
        late_aug=None,
        do_mixup=False,
        mixup_params={"prob": 0.5, "alpha": 1.0},
        n_cores=None,
        debug=False,
        df_filter_rule=None,
        do_noisereduce=False,
        late_normalize=False,
        use_sampler=False,
        shuffle=False,
        res_type="kaiser_best",
        pos_dtype=None,
        sampler_col=None,
        use_h5py=False,
    ):
        super().__init__()
        if use_h5py and precompute:
            raise ValueError("h5py files can not be used with `precompute`")
        if df is None and add_df_paths is None:
            raise ValueError("`df` OR/AND `add_df_paths` should be defined")
        if df is not None:
            df[f"{name_col}_with_root"] = None
        if add_df_paths is not None:
            cols_to_take = [
                target_col,
                sec_target_col,
                name_col,
                "duration_s",
            ]
            if rating_col is not None and rating_col not in cols_to_take:
                cols_to_take.append(rating_col)
            if sampler_col is not None and sampler_col not in cols_to_take:
                cols_to_take.append(sampler_col)
            # Create fake `df`
            if df is None:
                df = pd.DataFrame()
            else:
                df = df[cols_to_take]
            add_merged_df = pd.concat(
                [pd.read_csv(el)[cols_to_take] for el in add_df_paths],
                axis=0,
            ).reset_index(drop=True)
            if add_root is not None:
                add_merged_df[f"{name_col}_with_root"] = add_merged_df[
                    name_col
                ].apply(lambda x: pjoin(add_root, x))
            df = pd.concat([df, add_merged_df], axis=0).reset_index(drop=True)
        if df_filter_rule is not None:
            df = df_filter_rule(df)
        if debug:
            self.df = df.iloc[:1200]
        else:
            self.df = df
        self.df = self.df.reset_index(drop=True)
        if label_str2int_mapping_path is not None:
            self.label_str2int = load_json(label_str2int_mapping_path)
        else:
            self.label_str2int = None
        try:
            self.df["secondary_labels"] = self.df["secondary_labels"].apply(
                eval
            )
        except:
            print(
                "secondary_labels is not found in df. Maybe test or nocall mode"
            )
        if shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

        mask_col = self.df[f"{name_col}_with_root"].isna()
        self.df.loc[mask_col, f"{name_col}_with_root"] = self.df.loc[
            mask_col, name_col
        ].apply(lambda x: pjoin(root, x))
        if replace_pathes is not None:
            self.df[f"{name_col}_with_root"] = self.df[
                f"{name_col}_with_root"
            ].apply(
                lambda x: pjoin(
                    replace_pathes[1], relpath(x, replace_pathes[0])
                )
            )

        self.target_col = target_col
        self.sec_target_col = sec_target_col
        self.name_col = f"{name_col}_with_root"
        self.rating_col = rating_col
        self.late_normalize = late_normalize

        self.precompute = precompute
        self.use_h5py = use_h5py
        if self.use_h5py:
            self.df[self.name_col] = self.df[self.name_col].apply(
                lambda x: splitext(x)[0] + ".hdf5"
            )

        self.sample_rate = sample_rate
        self.do_noisereduce = do_noisereduce
        # save segment len in points (not in seconds)
        self.segment_len = int(self.sample_rate * segment_len)

        self.early_aug = early_aug
        self.late_aug = late_aug
        if mixup_params is not None and mixup_params.get("weights_path", None):
            if not use_sampler:
                raise ValueError(
                    "Mixup with weighted sampling requires `use_sampler=True`"
                )
        self.do_mixup = do_mixup
        self.mixup_params = mixup_params

        self.pos_dtype = pos_dtype
        self.res_type = res_type

        if self.precompute:
            if n_cores is not None:
                self.audio_cache = parallel_librosa_load(
                    audio_pathes=self.df[self.name_col].tolist(),
                    n_cores=n_cores,
                    return_sr=False,
                    sr=self.sample_rate,
                    do_normalize=not self.late_normalize,
                    do_noisereduce=do_noisereduce,
                    res_type=self.res_type,
                    pos_dtype=self.pos_dtype,
                )
                assert all(au is not None for au in self.audio_cache)
                self.audio_cache = {
                    i: el for i, el in enumerate(self.audio_cache)
                }
            else:
                print("NOT Parallel load")
                self.audio_cache = {
                    # Extract only audio, without sample_rate
                    i: load_pp_audio(
                        im_name,
                        sr=self.sample_rate,
                        do_noisereduce=do_noisereduce,
                        normalize=not self.late_normalize,
                        res_type=self.res_type,
                        pos_dtype=self.pos_dtype,
                    )
                    for i, im_name in tqdm(
                        enumerate(self.df[self.name_col].tolist()),
                        total=len(self.df),
                    )
                }

        if use_sampler:
            self.targets = (
                self.df[sampler_col].tolist()
                if sampler_col is not None
                else self.df[self.target_col].tolist()
            )
        if mixup_params.get("weights_path", None):
            self.weights = load_json(mixup_params["weights_path"])
            self.weights = torch.FloatTensor(
                [self.weights[el] for el in self.targets]
            )

    def turn_off_all_augs(self):
        print("All augs Turned Off")
        self.do_mixup = False
        self.early_aug = None
        self.late_aug = None

    def __len__(self):
        return len(self.df)

    def _prepare_sample_piece(self, input):
        if input.shape[0] < self.segment_len:
            pad_len = self.segment_len - input.shape[0]
            return np.pad(
                np.array(input) if self.use_h5py else input, ((pad_len, 0))
            )
        elif input.shape[0] > self.segment_len:
            start = np.random.randint(0, input.shape[0] - self.segment_len)
            return (
                np.array(input[start : start + self.segment_len])
                if self.use_h5py
                else input[start : start + self.segment_len]
            )
        else:
            return np.array(input) if self.use_h5py else input

    def _prepare_target(self, main_tgt, sec_tgt, all_labels=None):
        if all_labels is not None:
            if all_labels == "nocall":
                return torch.zeros(len(self.label_str2int)).float()
            else:
                all_tgt = [self.label_str2int[el] for el in all_labels.split()]
        else:
            if main_tgt == "nocall":
                all_tgt = []
            else:
                all_tgt = [self.label_str2int[main_tgt]] + [
                    self.label_str2int[el] for el in sec_tgt if el != ""
                ]
        all_tgt = torch.nn.functional.one_hot(
            torch.LongTensor(all_tgt), len(self.label_str2int)
        ).float()
        all_tgt = torch.clamp(all_tgt.sum(0), 0.0, 1.0)
        return all_tgt

    def _prepare_sample_target_from_idx(self, idx: int):
        if self.use_h5py:
            with h5py.File(self.df[self.name_col].iloc[idx], "r") as f:
                wave = self._prepare_sample_piece(f["au"])
        else:
            if self.precompute:
                wave = self.audio_cache[idx]
            else:
                # Extract only audio, without sample_rate
                wave = load_pp_audio(
                    self.df[self.name_col].iloc[idx],
                    sr=self.sample_rate,
                    do_noisereduce=self.do_noisereduce,
                    normalize=not self.late_normalize,
                    res_type=self.res_type,
                    pos_dtype=self.pos_dtype,
                )

            if self.pos_dtype is not None:
                wave = wave.astype(np.float32)

            wave = self._prepare_sample_piece(wave)

        main_tgt = self.df[self.target_col].iloc[idx]
        if self.sec_target_col is not None:
            sec_tgt = self.df[self.sec_target_col].iloc[idx]
        else:
            sec_tgt = [""]
        target = self._prepare_target(main_tgt, sec_tgt)
        if self.rating_col is not None:
            rating = self.df[self.rating_col].iloc[idx] / MAX_RATING
            assert 0.0 <= rating <= 1.0
            target = (target * rating).float()

        if self.early_aug is not None:
            raise RuntimeError("Not implemented")

        if self.late_normalize:
            wave = librosa.util.normalize(wave)

        return wave, target

    def _get_mixup_idx(self):
        if self.mixup_params.get("weights_path", None):
            mixup_idx = torch.multinomial(
                self.weights, 1, replacement=True
            ).item()
        else:
            mixup_idx = np.random.randint(0, self.__len__())
        return mixup_idx

    def __getitem__(self, index: int):
        wave, target = self._prepare_sample_target_from_idx(index)

        # Mixup/Cutmix/Fmix
        # .....
        if self.do_mixup and np.random.binomial(
            n=1, p=self.mixup_params["prob"]
        ):
            n_samples = self.mixup_params.get("n_samples", 1)
            assert n_samples >= 1
            if n_samples == 1:
                mixup_idx = self._get_mixup_idx()
                (
                    mixup_wave,
                    mixup_target,
                ) = self._prepare_sample_target_from_idx(mixup_idx)
                multimix = False
            else:
                n_samples = np.random.randint(1, n_samples + 1)
                mixup_wave, mixup_target = [], []
                for _ in range(n_samples):
                    mixup_idx = self._get_mixup_idx()
                    (
                        _mixup_wave,
                        _mixup_target,
                    ) = self._prepare_sample_target_from_idx(mixup_idx)
                    mixup_wave.append(_mixup_wave)
                    mixup_target.append(_mixup_target)
                multimix = True

            if self.mixup_params["alpha"] is None:
                if multimix:
                    wave = (sum(mixup_wave) + wave) / (n_samples + 1)
                    target = sum(mixup_target) + target
                else:
                    wave = (mixup_wave + wave) / 2
                    target = mixup_target + target
            else:
                mix_weight = np.random.beta(
                    self.mixup_params["alpha"], self.mixup_params["alpha"]
                )
                if self.mixup_params.get("weight_trim", False):
                    mix_weight = min(
                        max(mix_weight, self.mixup_params["weight_trim"][0]),
                        self.mixup_params["weight_trim"][1],
                    )
                wave = mix_weight * mixup_wave + (1 - mix_weight) * wave
                if self.mixup_params.get("hard_target", True):
                    target = mixup_target + target
                else:
                    target = (
                        mix_weight * mixup_target + (1 - mix_weight) * target
                    )

            target = torch.clamp(target, min=0, max=1.0)
            if self.late_normalize:
                wave = librosa.util.normalize(wave)

        if self.late_aug is not None:
            wave = self.late_aug(wave)

        return wave, target


class WaveAllFileDataset(WaveDataset):
    def __init__(
        self,
        df,
        root,
        label_str2int_mapping_path,
        df_path=None,
        add_df_paths=None,
        target_col="primary_label",
        sec_target_col="secondary_labels",
        all_target_col="birds",
        name_col="filename",
        duration_col="duration_s",
        sample_id="sample_id",
        sample_rate=32_000,
        segment_len=5.0,
        step=None,
        lookback=None,
        lookahead=None,
        precompute=False,
        early_aug=None,
        late_aug=None,
        do_mixup=False,
        mixup_params={"prob": 0.5, "alpha": 1.0},
        n_cores=None,
        debug=False,
        df_filter_rule=None,
        use_audio_cache=False,
        verbose=True,
        test_mode=False,
        soundscape_mode=False,
        use_eps_in_slicing=False,
        dfidx_2_sample_id=False,
        do_noisereduce=False,
        late_normalize=False,
        use_h5py=False,
        # In BirdClef Comp, it is claimed that all samples in 32K sr
        # we will just validate it, without doing resampling
        validate_sr=None,
        **kwargs,
    ):
        if kwargs:
            warnings.warn(
                f"WaveAllFileDataset received extra parameters: {kwargs}"
            )
        if df_path is not None:
            df = pd.read_csv(df_path)
        if test_mode and soundscape_mode:
            raise RuntimeError(
                "only test_mode or soundscape_mode can be activated"
            )
        if precompute and use_audio_cache:
            raise RuntimeError("audio_cache is useless if you use precompute")
        super().__init__(
            df=df,
            add_df_paths=add_df_paths,
            root=root,
            label_str2int_mapping_path=label_str2int_mapping_path,
            target_col=target_col,
            sec_target_col=sec_target_col,
            name_col=name_col,
            sample_rate=sample_rate,
            segment_len=segment_len,
            # In case of soundscape_mode, cache will be computed in another way
            precompute=precompute and not soundscape_mode,
            early_aug=early_aug,
            late_aug=late_aug,
            do_mixup=do_mixup,
            mixup_params=mixup_params,
            n_cores=n_cores,
            debug=debug,
            df_filter_rule=df_filter_rule,
            do_noisereduce=do_noisereduce,
            late_normalize=late_normalize,
            use_h5py=use_h5py,
        )
        self.validate_sr = validate_sr
        if precompute and soundscape_mode:
            self.audio_cache = {
                # Extract only audio, without sample_rate
                im_name: load_pp_audio(
                    im_name,
                    sr=None
                    if self.validate_sr is not None
                    else self.sample_rate,
                    do_noisereduce=do_noisereduce,
                    normalize=not self.late_normalize,
                    validate_sr=self.validate_sr,
                )
                for im_name in tqdm(
                    set(self.df[self.name_col]),
                    total=len(set(self.df[self.name_col])),
                )
            }
            self.precompute = True

        self.duration_col = duration_col
        self.verbose = verbose
        self.test_mode = test_mode
        self.soundscape_mode = soundscape_mode
        self.all_target_col = all_target_col
        self.sample_id = sample_id
        self.dfidx_2_sample_id = dfidx_2_sample_id
        eps = EPS if use_eps_in_slicing else 0

        if sample_id is not None:
            self.df[self.sample_id] = (
                self.df[self.sample_id].astype("category").cat.codes
            )

        self.sampleidx_2_dfidx = {}
        if lookahead is not None and lookback is not None:
            print("Dataset in hard_slicing mode")
            if step is None:
                step = segment_len
            self.hard_slicing = True
            itter = 0
            if soundscape_mode:
                samples_generator = enumerate(
                    self.df.drop_duplicates(self.name_col)[self.duration_col]
                )
            else:
                samples_generator = enumerate(self.df[self.duration_col])
            for dfidx, dur in samples_generator:
                real_start = -lookback
                while real_start + lookback < dur + eps:
                    self.sampleidx_2_dfidx[itter] = {
                        "dfidx": itter if soundscape_mode else dfidx,
                        "start": int(real_start * self.sample_rate),
                        "end": int(
                            (real_start + lookback + segment_len + lookahead)
                            * self.sample_rate
                        ),
                        "end_s": min(
                            int(real_start + lookback + segment_len), dur
                        ),
                    }
                    real_start += step
                    itter += 1
        else:
            self.hard_slicing = False
            t_start = 0
            if soundscape_mode:
                samples_generator = enumerate(
                    self.df.drop_duplicates(self.name_col)[self.duration_col]
                )
            else:
                samples_generator = enumerate(self.df[self.duration_col])
            for dfidx, dur in samples_generator:
                n_pieces_in_file = math.ceil(dur / segment_len)
                self.sampleidx_2_dfidx.update(
                    {
                        i
                        + t_start: {
                            "dfidx": i + t_start if soundscape_mode else dfidx,
                            "start": int(segment_len * i * self.sample_rate),
                            "end_s": int(segment_len * (i + 1)),
                        }
                        for i in range(n_pieces_in_file)
                    }
                )
                t_start += n_pieces_in_file

        self.use_audio_cache = use_audio_cache
        self.test_audio_cache = {"au": None, "dfidx": None}

    def _print_v(self, msg):
        if self.verbose:
            print(msg)

    def _hadle_au_cache(self, dfidx):
        if (
            self.test_audio_cache["dfidx"] is None
            or self.test_audio_cache["dfidx"] != dfidx
        ):
            start_time = time()
            self._print_v(
                f"Loading {self.df[self.name_col].iloc[dfidx]} to audio cache"
            )
            self.test_audio_cache["au"] = load_pp_audio(
                self.df[self.name_col].iloc[dfidx],
                sr=None if self.validate_sr is not None else self.sample_rate,
                do_noisereduce=self.do_noisereduce,
                normalize=not self.late_normalize,
                validate_sr=self.validate_sr,
            )
            self.test_audio_cache["dfidx"] = dfidx
            self._print_v(f"Loading took {time() - start_time} seconds")
        return self.test_audio_cache["au"]

    def __len__(self):
        return len(self.sampleidx_2_dfidx)

    def _prepare_sample_piece_hard(self, input, start, end):
        # Process right pad or end trim
        if end > input.shape[0]:
            input = np.pad(
                np.array(input) if self.use_h5py else input,
                ((0, end - input.shape[0])),
            )
        else:
            input = np.array(input[:end]) if self.use_h5py else input[:end]
        # Process left pad or start trim
        if start < 0:
            input = np.pad(input, ((-start, 0)))
        else:
            input = input[start:]

        return input

    def _prepare_sample_piece(self, input, start):
        input = (
            np.array(input[start : start + self.segment_len])
            if self.use_h5py
            else input[start : start + self.segment_len]
        )

        if input.shape[0] < self.segment_len:
            pad_len = self.segment_len - input.shape[0]
            input = np.pad(input, ((pad_len, 0)))
        else:
            pad_len = 0

        return input, pad_len

    def _prepare_sample_target_from_idx(self, idx: int):
        map_dict = self.sampleidx_2_dfidx[idx]
        dfidx = map_dict["dfidx"]
        start = map_dict["start"]
        end = start + self.segment_len

        if self.use_h5py:
            with h5py.File(self.df[self.name_col].iloc[dfidx], "r") as f:
                if self.hard_slicing:
                    wave = self._prepare_sample_piece_hard(
                        f["au"], start=start, end=map_dict["end"]
                    )
                else:
                    wave, _ = self._prepare_sample_piece(f["au"], start=start)
        else:
            if self.precompute:
                if self.soundscape_mode:
                    wave = self.audio_cache[self.df[self.name_col].iloc[dfidx]]
                else:
                    wave = self.audio_cache[dfidx]
            else:
                # Extract only audio, without sample_rate
                if self.use_audio_cache:
                    wave = self._hadle_au_cache(dfidx)
                else:
                    wave = load_pp_audio(
                        self.df[self.name_col].iloc[dfidx],
                        sr=None
                        if self.validate_sr is not None
                        else self.sample_rate,
                        do_noisereduce=self.do_noisereduce,
                        normalize=not self.late_normalize,
                        validate_sr=self.validate_sr,
                    )
            if self.hard_slicing:
                wave = self._prepare_sample_piece_hard(
                    wave, start=start, end=map_dict["end"]
                )
            else:
                wave, _ = self._prepare_sample_piece(wave, start=start)

        if self.test_mode:
            target = -1
        else:
            if self.soundscape_mode:
                target = self._prepare_target(
                    main_tgt=None,
                    sec_tgt=None,
                    all_labels=self.df[self.all_target_col].iloc[dfidx],
                )
            else:
                main_tgt = self.df[self.target_col].iloc[dfidx]
                sec_tgt = self.df[self.sec_target_col].iloc[dfidx]
                target = self._prepare_target(main_tgt, sec_tgt)

        if self.dfidx_2_sample_id:
            dfidx = self.df[self.sample_id].iloc[dfidx]

        end = map_dict["end_s"]

        if self.early_aug is not None:
            raise RuntimeError("Not implemented")

        if self.late_normalize:
            wave = librosa.util.normalize(wave)

        return wave, target, dfidx, start, end

    def __getitem__(self, index: int):
        wave, target, dfidx, start, end = self._prepare_sample_target_from_idx(
            index
        )

        # Mixup/Cutmix/Fmix
        # .....
        if self.do_mixup and np.random.binomial(
            n=1, p=self.mixup_params["prob"]
        ):
            raise RuntimeError("Not implemented")

        if self.late_aug is not None:
            raise RuntimeError("Not implemented")

        return wave, target, dfidx, start, end
