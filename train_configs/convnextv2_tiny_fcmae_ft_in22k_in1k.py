from glob import glob

import torch
from catalyst import callbacks

from code_base.augmentations.transforms import BackgroundNoise
from code_base.callbacks import PaddedCMAPScore, SWACallback
from code_base.datasets import WaveAllFileDataset, WaveDataset
from code_base.forwards import MultilabelClsForwardLongShort
from code_base.models import RandomFiltering, WaveCNNAttenClasifier
from code_base.train_functions import catalyst_training

B_S = 64
TRAIN_PERIOD = 5.0
N_EPOCHS = 50
ROOT_PATH = "data/birdclef_2023/train_audio"
LATE_NORMALIZE = True

PATH_TO_JSON_MAPPING = "data/bird2int_2023.json"
PRECOMPUTE = True
DEBUG = False

CONFIG = {
    "seed": 1243,
    "df_path": "data/train_metadata_extended.csv",
    "split_path": None,
    "exp_name": f"convnextv2_tiny_fcmae_ft_in22k_in1k_384_Exp_noamp_64bs_5sec_mixupP05_RandomFiltering_balancedSampler_lr1e4_CosineLREpoch50_ValV2_202xXcAddDataNoAddSecLabelsMayXCV1_BackGroundSoundScapeP05_FocalLoss_DPR02_noval",
    "files_to_save": (
        glob("code_base/**/*.py") + [__file__] + ["scripts/main_train.py"]
    ),
    "folds": None,
    "train_function": catalyst_training,
    "train_function_args": {
        "train_dataset_class": WaveDataset,
        "train_dataset_config": {
            "root": ROOT_PATH,
            "label_str2int_mapping_path": PATH_TO_JSON_MAPPING,
            "precompute": PRECOMPUTE,
            "n_cores": 32,
            "debug": DEBUG,
            "do_mixup": True,
            "mixup_params": {"prob": 0.5, "alpha": None},
            "segment_len": TRAIN_PERIOD,
            "late_normalize": LATE_NORMALIZE,
            "shuffle": True,
            "sampler_col": "primary_label",
            "add_df_paths": [
                "data/train_metadata_extended_2020_2022_no2023_scored_nodupl_v1_only_prime_2023SecLabels.csv",
                "data/scored_2023_xc_2023SecLabels.csv",
                "data/scored_2023_xc_2023SecLabels_till_12_05_2023_v2.csv",
            ],
            "add_root": "data/audio",
            "late_aug": BackgroundNoise(
                p=0.5,
                esc50_root="data/soundscapes_nocall/train_audio",
                esc50_df_path="data/v1_no_call_meta.csv",
                normalize=LATE_NORMALIZE,
            ),
        },
        "val_dataset_class": None,
        "val_dataset_config": None,
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
            backbone="convnextv2_tiny.fcmae_ft_in22k_in1k_384",
            mel_spec_paramms={
                "sample_rate": 32000,
                "n_mels": 128,
                "f_min": 20,
                "n_fft": 2048,
                "hop_length": 512,
                "normalized": True,
            },
            head_config={
                "p": 0.5,
                "num_class": 264,
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
            ),
            callbacks.SchedulerCallback(
                mode="epoch",
            ),
        ],
        "valid_loader": "train",
        "main_metric": "loss",
        "minimize_metric": True,
        "check": False,
        "use_amp": False,
        "label_str2int_path": PATH_TO_JSON_MAPPING,
        "class_weights_path": "data/xc_birds_202x_only_scored_sample_weights_v1_till_12_05_2023_v2.json",
        "use_sampler": True,
    },
}
