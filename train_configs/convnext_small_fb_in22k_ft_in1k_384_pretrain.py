from glob import glob

import torch
from catalyst import callbacks

from code_base.augmentations.transforms import BackgroundNoise, OneOf
from code_base.callbacks import PaddedCMAPScore, SWACallback
from code_base.datasets import WaveAllFileDataset, WaveDataset
from code_base.forwards import MultilabelClsForwardLongShort
from code_base.models import RandomFiltering, WaveCNNAttenClasifier
from code_base.train_functions import catalyst_training

B_S = 64
TRAIN_PERIOD = 5.0
N_EPOCHS = 50
ROOT_PATH = ""
LATE_NORMALIZE = True

PATH_TO_JSON_MAPPING = "data/bird2id_v3.json"
DEBUG = False

CONFIG = {
    "seed": 1243,
    "df_path": "data/train_metadata_extended_shorten_and_2023x_v3.csv",
    "split_path": "data/add_xeno_canto_old_pretrain_df_oof_split_v3.npy",
    "exp_name": f"convnext_small_fb_in22k_ft_in1k_384_Exp_noamp_64bs_5sec_mixupP05_RandomFiltering_balancedSampler_lr1e4_CosineLREpoch50_202xXcAddDataNoAddSecLabelsMayXCV1_BackGroundSoundScapeORESC50P05_SpecAugV1_FocalLoss_noval_PretrainV3",
    "files_to_save": (
        glob("code_base/**/*.py") + [__file__] + ["scripts/main_train.py"]
    ),
    "folds": 0,
    "train_function": catalyst_training,
    "train_function_args": {
        "train_dataset_class": WaveDataset,
        "train_dataset_config": {
            "root": ROOT_PATH,
            "label_str2int_mapping_path": PATH_TO_JSON_MAPPING,
            "precompute": False,
            "n_cores": 32,
            "debug": DEBUG,
            "do_mixup": True,
            "mixup_params": {"prob": 0.5, "alpha": None},
            "segment_len": TRAIN_PERIOD,
            "late_normalize": LATE_NORMALIZE,
            "shuffle": True,
            "sampler_col": "primary_label",
            "use_h5py": True,
            "name_col": "filename_h5py",
            "late_aug": OneOf(
                [
                    BackgroundNoise(
                        p=0.5,
                        esc50_root="data/soundscapes_nocall/train_audio",
                        esc50_df_path="data/v1_no_call_meta.csv",
                        normalize=LATE_NORMALIZE,
                    ),
                    BackgroundNoise(
                        p=0.5,
                        esc50_root="data/esc50/audio",
                        esc50_df_path="data/esc50_background.csv",
                        esc50_cats_to_include=[
                            "dog",
                            "rain",
                            "insects",
                            "hen",
                            "engine",
                            "hand_saw",
                            "pig",
                            "rooster",
                            "sea_waves",
                            "cat",
                            "crackling_fire",
                            "thunderstorm",
                            "chainsaw",
                            "train",
                            "sheep",
                            "wind",
                            "footsteps",
                            "frog",
                            "cow",
                            "crickets",
                        ],
                        normalize=LATE_NORMALIZE,
                    ),
                ]
            ),
        },
        "val_dataset_class": WaveAllFileDataset,
        "val_dataset_config": {
            "root": ROOT_PATH,
            "label_str2int_mapping_path": PATH_TO_JSON_MAPPING,
            "precompute": False,
            "n_cores": 32,
            "debug": DEBUG,
            "segment_len": 5,
            "sample_id": None,
            "late_normalize": LATE_NORMALIZE,
            "use_h5py": True,
            "name_col": "filename_h5py",
        },
        "train_dataloader_config": {
            "batch_size": B_S,
            # "shuffle": True,
            "drop_last": True,
            "num_workers": 8,
            "pin_memory": True,
        },
        "val_dataloader_config": {
            "batch_size": B_S,
            "shuffle": False,
            "drop_last": False,
            "num_workers": 8,
            "pin_memory": True,
        },
        "nn_model_class": WaveCNNAttenClasifier,
        "nn_model_config": dict(
            backbone="convnext_small.fb_in22k_ft_in1k_384",
            mel_spec_paramms={
                "sample_rate": 32000,
                "n_mels": 128,
                "f_min": 20,
                "n_fft": 2048,
                "hop_length": 512,
                "normalized": True,
            },
            spec_augment_config={
                "freq_mask": {
                    "mask_max_length": 10,
                    "mask_max_masks": 3,
                    "p": 0.3,
                    "inplace": True,
                },
                "time_mask": {
                    "mask_max_length": 20,
                    "mask_max_masks": 3,
                    "p": 0.3,
                    "inplace": True,
                },
            },
            head_config={
                "p": 0.5,
                "num_class": 822,
                "train_period": TRAIN_PERIOD,
                "infer_period": TRAIN_PERIOD,
            },
            exportable=True,
        ),
        "optimizer_init": lambda model: torch.optim.Adam(
            model.parameters(), lr=1e-4
        ),
        "scheduler_init": lambda optimizer, len_train: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=N_EPOCHS, T_mult=1, eta_min=1e-6, last_epoch=-1
        ),
        "forward": MultilabelClsForwardLongShort(
            loss_type="baseline",
            use_weights=False,
            long_files_loaders=("valid",),
            batch_aug=RandomFiltering(
                min_db=-20, is_wave=True, normalize_wave=LATE_NORMALIZE
            ),
            use_focal_loss=True,
        ),
        "n_epochs": N_EPOCHS,
        "catalyst_callbacks": lambda: [
            callbacks.BackwardCallback(metric_key="loss"),
            callbacks.OptimizerCallback(
                metric_key="loss", accumulation_steps=1
            ),
            PaddedCMAPScore(
                loader_names=("valid",),
                aggr_key="dfidx",
                output_long_key=None,
                output_key="clipwise_pred_long",
                use_sigmoid=False,
                label_str2int_mapping_path=PATH_TO_JSON_MAPPING,
                scored_bird_path="data/scored_birds_xc_pretrain_val_v3.json",
            ),
            callbacks.SchedulerCallback(
                mode="epoch",
            ),
        ],
        "valid_loader": "valid",
        "main_metric": "padded_cmap",
        "minimize_metric": False,
        "check": False,
        "use_amp": False,
        "label_str2int_path": PATH_TO_JSON_MAPPING,
        "class_weights_path": "data/sample_weights_v3.json",
        "use_sampler": True,
    },
}
