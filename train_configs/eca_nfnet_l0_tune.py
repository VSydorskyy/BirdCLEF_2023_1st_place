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
ROOT_PATH = "/home/vova/data/exps/BirdCLEF_2023/birdclef_2023/train_audio/"
LATE_NORMALIZE = True

PATH_TO_JSON_MAPPING = (
    "/home/vova/data/exps/BirdCLEF_2023/final_pretrains/bird2id_v3.json"
)
PRECOMPUTE = True
DEBUG = False

CONFIG = {
    "seed": 1243,
    "df_path": "/home/vova/data/exps/BirdCLEF_2023/birdclef_2023/train_metadata_extended.csv",
    "split_path": None,
    "exp_name": f"eca_nfnet_l0_Exp_noamp_64bs_5sec_mixupP05_RandomFiltering_balancedSampler_lr1e3_CosineLREpoch50_ValV2_202xXcAddDataNoAddSecLabels_BackGroundSoundScapeP05_SpecAugV1_FocalLoss_DPR02_TuneV3_noval",
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
                "/home/vova/data/exps/BirdCLEF_2023/train_metadata_extended_2020_2022_no2023_scored_nodupl_v1_only_prime_2023SecLabels.csv",
                "/home/vova/data/exps/BirdCLEF_2023/xeno_canto/scored_2023_xc_2023SecLabels.csv",
            ],
            "late_aug": BackgroundNoise(
                p=0.5,
                esc50_root="",
                esc50_df_path="/home/vova/data/exps/BirdCLEF_2023/soundscapes_processed/v1_no_call_meta.csv",
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
        "val_dataloader_config": None,
        "nn_model_class": WaveCNNAttenClasifier,
        "nn_model_config": dict(
            backbone="eca_nfnet_l0",
            add_backbone_config={"drop_path_rate": 0.2},
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
            model.parameters(), lr=1e-3
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
        "class_weights_path": "/home/vova/data/exps/BirdCLEF_2023/xc_birds_202x_only_scored_sample_weights_v1.json",
        "use_sampler": True,
        "pretrained_chekpoints": "logdirs/eca_nfnet_l0_Exp_noamp_64bs_5sec_mixupP05_RandomFiltering_balancedSampler_lr1e3_CosineLREpoch50_202xXcAddDataNoAddSecLabelsMayXCV1_BackGroundSoundScapeORESC50P05_SpecAugV1_FocalLoss_noval_PretrainV3/checkpoints/model.last.pth",
        "checkpoint_type": "catalyst",
        "use_one_checkpoint": True,
    },
}
