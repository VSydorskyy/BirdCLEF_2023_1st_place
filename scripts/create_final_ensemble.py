import os
from copy import deepcopy

import numpy as np
import torch

from code_base.models import WaveCNNAttenClasifier
from code_base.utils import load_json
from code_base.utils.inference_utils import apply_avarage_weights_on_swa_path
from code_base.utils.onnx_utils import ONNXEnsemble, convert_to_onnx


def create_model_and_upload_chkp(
    model_class,
    model_config,
    model_device,
    model_chkp,
    use_distributed=False,
    swa_checkpoint=None,
    strict=True,
):
    print(f"Processing chkp {model_chkp}")
    if "swa" in model_chkp:
        print(
            "swa by {}".format(
                os.path.splitext(os.path.basename(model_chkp))[0]
            )
        )
        t_chkp = apply_avarage_weights_on_swa_path(
            model_chkp,
            use_distributed=use_distributed,
            take_best=swa_checkpoint,
        )
    else:
        print("vanilla model")
        t_chkp = torch.load(model_chkp, map_location="cpu")

    t_model = model_class(**model_config, device=model_device)
    if not strict:
        t_chkp_keys = set(t_chkp.keys())
        t_model_keys = set(t_model.state_dict().keys())
        print(f"Missing keys in t_chkp: {t_model_keys - t_chkp_keys}")
        print(f"Missing keys in t_model: {t_chkp_keys - t_model_keys}")
    t_model.load_state_dict(t_chkp, strict=strict)
    t_model.eval()
    return t_model


def main():
    bird2id_2023 = load_json("data/bird2int_2023.json")
    bird2id_202x = load_json("data/bird2id_v3.json")

    id2bird_2023 = {v: k for k, v in bird2id_2023.items()}

    rearrange_indices = np.array(
        [bird2id_202x[id2bird_2023[i]] for i in range(len(bird2id_2023))]
    ).astype(int)

    MODEL_CLASS = WaveCNNAttenClasifier
    TRAIN_PERIOD = 5
    STRICT_LOAD = False

    MODELS = [
        {
            "model_config": dict(
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
                head_config={
                    "p": 0.5,
                    "num_class": 822,
                    "train_period": TRAIN_PERIOD,
                    "infer_period": TRAIN_PERIOD,
                    "output_type": "clipwise_timewisemax_pred_short",
                    "infer_framewise_max_coef": 0.25,
                },
                exportable=True,
                selected_indices=rearrange_indices,
            ),
            "exp_name": "eca_nfnet_l0_Exp_noamp_64bs_5sec_mixupP05_RandomFiltering_balancedSampler_lr1e3_CosineLREpoch50_ValV2_202xXcAddDataNoAddSecLabels_BackGroundSoundScapeP05_SpecAugV1_FocalLoss_DPR02_TuneV3_noval",
            "fold": None,
            "chkp_name": "model.last.pth",
            "swa_checkpoint": None,
            "distributed_chkp": False,
        },
        {
            "model_config": dict(
                backbone="convnext_small.fb_in22k_ft_in1k_384",
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
                    "num_class": 822,
                    "train_period": TRAIN_PERIOD,
                    "infer_period": TRAIN_PERIOD,
                    "output_type": "clipwise_timewisemax_pred_short",
                    "infer_framewise_max_coef": 0.25,
                },
                exportable=True,
                selected_indices=rearrange_indices,
            ),
            "exp_name": "convnext_small_fb_in22k_ft_in1k_384_Exp_noamp_64bs_5sec_mixupP05_RandomFiltering_balancedSampler_lr1e4_CosineLREpoch50_ValV2_202xXcAddDataNoAddSecLabelsMayXCV1_BackGroundSoundScapeP05_SpecAugV1_FocalLoss_TuneV3_noval",
            "fold": None,
            "chkp_name": "model.last.pth",
            "swa_checkpoint": None,
            "distributed_chkp": False,
        },
        {
            "model_config": dict(
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
                    "output_type": "clipwise_timewisemax_pred_short",
                    "infer_framewise_max_coef": 0.25,
                },
                exportable=True,
            ),
            "exp_name": "convnextv2_tiny_fcmae_ft_in22k_in1k_384_Exp_noamp_64bs_5sec_mixupP05_RandomFiltering_balancedSampler_lr1e4_CosineLREpoch50_ValV2_202xXcAddDataNoAddSecLabelsMayXCV1_BackGroundSoundScapeP05_FocalLoss_DPR02_noval",
            "fold": None,
            "chkp_name": "model.last.pth",
            "swa_checkpoint": None,
            "distributed_chkp": False,
        },
    ]

    model = [
        create_model_and_upload_chkp(
            model_class=MODEL_CLASS,
            model_config=config["model_config"],
            model_device="cuda",
            model_chkp=(
                f"logdirs/{config['exp_name']}/fold_{config['fold']}/checkpoints/{config['chkp_name']}"
                if config["fold"] is not None
                else f"logdirs/{config['exp_name']}/checkpoints/{config['chkp_name']}"
            ),
            swa_checkpoint=config["swa_checkpoint"],
            use_distributed=config["distributed_chkp"],
            strict=STRICT_LOAD,
        )
        for config in MODELS
    ]

    exportable_ensem = ONNXEnsemble(
        model_class=MODEL_CLASS,
        configs=[deepcopy(config["model_config"]) for config in MODELS],
        avarage_type="gaus",
    )
    for model_id in range(len(exportable_ensem.models)):
        exportable_ensem.models[model_id].load_state_dict(
            model[model_id].state_dict()
        )
    exportable_ensem.eval()
    convert_to_onnx(
        model_to_convert=exportable_ensem,
        sample_input=torch.randn(5, TRAIN_PERIOD * 32_000),
        base_path=f"logdirs/convnext_small_fb_in22k_ft_in1k_384__convnextv2_tiny_fcmae_ft_in22k_in1k_384__eca_nfnet_l0_noval_v32_075Clipwise025TimeMax_GausMean/onnx_ensem",
    )


if __name__ == "__main__":
    main()
