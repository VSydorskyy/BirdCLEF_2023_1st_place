from copy import deepcopy
from importlib.util import spec_from_file_location
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import timm
import torch
import torch.nn as nn
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram

try:
    from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
except:
    print("`speechbrain` was not imported")
try:
    from nnAudio.Spectrogram import CQT1992v2
except:
    print("`nnAudioSTFT` was not imported")
try:
    import leaf_audio_pytorch.frontend as frontend
except:
    print("`LEAF` was not imported")

from ..augmentations.spec_augment import CustomFreqMasking, CustomTimeMasking
from .blocks import (
    AttHead,
    Clasifier,
    NormalizeMelSpec,
    PoolingLayer,
    QuantizableAmplitudeToDB,
    TraceableMelspec,
)


class WaveCNNClasifier(nn.Module):
    def __init__(
        self,
        backbone: str,
        device: str,
        mel_spec_paramms: Dict[str, Any],
        classifiier_config: Dict[str, Any],
        spec_extractor: str = "Melspec",
        exportable: bool = False,
        quantizable: bool = False,
        pool_type: str = "AdaptiveAvgPool2d",
        top_db: float = 80.0,
        pretrained: bool = True,
        train_time: Optional[float] = None,
        test_time: Optional[float] = None,
        inference_mode: bool = False,
        add_backbone_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        if train_time is not None and test_time is not None:
            assert train_time > test_time and train_time % test_time == 0
            # self.use_train_test_diff_len = True
            self.trainx = int(train_time // test_time)
            print(f"Trainx = {self.trainx}")
        else:
            self.trainx = None

        self.device = device
        self.inference_mode = inference_mode
        self.logmelspec_extractor = self._create_feature_extractor(
            mel_spec_paramms, exportable, top_db, quantizable, spec_extractor
        )
        add_backbone_config = (
            {} if add_backbone_config is None else add_backbone_config
        )
        self.backbone = timm.create_model(
            backbone,
            features_only=True,
            pretrained=pretrained,
            in_chans=1,
            **add_backbone_config,
        )
        self.pool = PoolingLayer(pool_type)
        self.classifier = Clasifier(
            nn_embed_size=self.backbone.feature_info.channels()[-1],
            **classifiier_config,
        )
        self.sigmoid = nn.Sigmoid()
        self.to(self.device)

    def _create_feature_extractor(
        self, mel_spec_paramms, exportable, top_db, quantizable, spec_extractor
    ):
        if spec_extractor == "Melspec":
            if exportable:
                spec_init = TraceableMelspec
            else:
                spec_init = MelSpectrogram
        elif spec_extractor == "CQT":
            spec_init = CQT1992v2
        else:
            raise NotImplementedError(f"{spec_extractor} not implemented")
        if isinstance(mel_spec_paramms, list):
            self._n_specs = len(mel_spec_paramms)
            return nn.ModuleList(
                [
                    nn.Sequential(
                        spec_init(**mel_spec_paramm, quantizable=True)
                        if quantizable
                        else spec_init(**mel_spec_paramm),
                        QuantizableAmplitudeToDB(top_db=top_db)
                        if quantizable
                        else AmplitudeToDB(top_db=top_db),
                        NormalizeMelSpec(exportable=exportable),
                    )
                    for mel_spec_paramm in mel_spec_paramms
                ]
            )
        else:
            self._n_specs = 1
            return nn.Sequential(
                spec_init(**mel_spec_paramms, quantizable=True)
                if quantizable
                else spec_init(**mel_spec_paramms),
                QuantizableAmplitudeToDB(top_db=top_db)
                if quantizable
                else AmplitudeToDB(top_db=top_db),
                NormalizeMelSpec(exportable=exportable),
            )

    def forward(self, input, return_spec_feature=False, return_cnn_emb=False):
        spec = self.logmelspec_extractor(input)

        if self.trainx is not None and self.training:
            spec_len = spec.shape[2]
            bs_size = spec.shape[0]
            slice_len = spec_len % self.trainx
            if slice_len > 0:
                spec = spec[:, :, : -int(slice_len)]
                spec_len = spec.shape[2]
            spec_piece_len = int(spec_len // self.trainx)
            spec = torch.cat(
                [
                    spec[:, :, i * spec_piece_len : (i + 1) * spec_piece_len]
                    for i in range(self.trainx)
                ],
                axis=0,
            )

        if return_spec_feature:
            return spec
        emb = self.backbone(spec[:, None])[-1]

        if self.trainx is not None and self.training:
            assert emb.shape[0] % bs_size == 0
            emb = torch.cat(
                [
                    emb[
                        i * bs_size : (i + 1) * bs_size,
                        :,
                    ]
                    for i in range(emb.shape[0] // bs_size)
                ],
                axis=3,
            )

        emb = self.pool(emb)
        if return_cnn_emb:
            return emb
        logits = self.classifier(emb)

        if self.inference_mode:
            return self.sigmoid(logits)
        else:
            return {
                "clipwise_logits_long": logits,
                "clipwise_pred_long": self.sigmoid(logits),
            }


class WaveCNNAttenClasifier(nn.Module):
    def __init__(
        self,
        backbone: Optional[str],
        device: str,
        mel_spec_paramms: Dict[str, Any],
        head_config: Optional[Dict[str, Any]],
        transformer_backbone: bool = False,
        head_type: str = "AttHead",
        top_db: float = 80.0,
        pretrained: bool = True,
        first_conv_name: str = "conv_stem",
        first_conv_stride_overwrite: Optional[
            Union[int, Tuple[int, int]]
        ] = None,
        exportable: bool = False,
        quantizable: bool = False,
        central_crop_input: Optional[float] = None,
        selected_indices: Optional[List[int]] = None,
        use_sigmoid: bool = False,
        spec_extractor: str = "Melspec",
        add_backbone_config: Optional[Dict[str, Any]] = None,
        deep_supervision_steps: Optional[int] = None,
        permute_backbone_emb: Optional[Tuple[int]] = None,
        no_head_inference_mode: bool = False,
        spec_augment_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        add_backbone_config_ = deepcopy(add_backbone_config)
        head_config_ = deepcopy(head_config)
        mel_spec_paramms_ = deepcopy(mel_spec_paramms)

        self.device = device
        self.central_crop_input = central_crop_input
        self.use_selected_indices = selected_indices is not None
        self.quantizable = quantizable
        self.deep_supervision_steps = deep_supervision_steps
        self.transformer_backbone = transformer_backbone
        self.permute_backbone_emb = permute_backbone_emb
        self.no_head_inference_mode = no_head_inference_mode
        self.logmelspec_extractor = self._create_feature_extractor(
            mel_spec_paramms_, exportable, top_db, quantizable, spec_extractor
        )
        if spec_augment_config is not None:
            self.spec_augment = []
            if "freq_mask" in spec_augment_config:
                self.spec_augment.append(
                    CustomFreqMasking(**spec_augment_config["freq_mask"])
                )
            if "time_mask" in spec_augment_config:
                self.spec_augment.append(
                    CustomTimeMasking(**spec_augment_config["time_mask"])
                )
            self.spec_augment = nn.Sequential(*self.spec_augment)
        else:
            self.spec_augment = None
        if backbone is not None:
            add_backbone_config_ = (
                {} if add_backbone_config_ is None else add_backbone_config_
            )
            if self.transformer_backbone:
                head_dropout = add_backbone_config_.pop("head_dropout", 0.0)
                self.backbone = timm.create_model(
                    backbone,
                    pretrained=pretrained,
                    exportable=exportable,
                    in_chans=self._n_specs,
                    **add_backbone_config_,
                )
                self.backbone.head.drop.p = head_dropout
            else:
                self.backbone = timm.create_model(
                    backbone,
                    features_only=True,
                    pretrained=pretrained,
                    exportable=exportable,
                    in_chans=self._n_specs,
                    **add_backbone_config_,
                )
        if first_conv_stride_overwrite is not None:
            if isinstance(first_conv_name, str):
                first_conv_name = [first_conv_name]
            for conv_name in first_conv_name:
                setattr(
                    getattr(self.backbone, conv_name),
                    "stride",
                    first_conv_stride_overwrite,
                )
        if head_config_ is not None and not self.transformer_backbone:
            if head_type == "AttHead":
                backbone_channels = self.backbone.feature_info.channels()
                self.head = AttHead(
                    in_chans=(
                        backbone_channels[-1]
                        if deep_supervision_steps is None
                        else backbone_channels[-deep_supervision_steps:]
                    ),
                    exportable=exportable,
                    **head_config_,
                )
            elif head_type == "Clasifier":
                self.head = nn.Sequential(
                    PoolingLayer(
                        pool_type=head_config_.pop(
                            "pool_type", "AdaptiveAvgPool2d"
                        )
                    ),
                    Clasifier(
                        nn_embed_size=self.backbone.feature_info.channels()[
                            -1
                        ],
                        **head_config_,
                    ),
                )
            else:
                raise NotImplementedError(f"{head_type} not implemented")
        else:
            self.head = None
        if self.use_selected_indices:
            self.register_buffer(
                "selected_indices", torch.LongTensor(selected_indices)
            )
        if use_sigmoid or self.transformer_backbone:
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = None
        self.to(self.device)

    def _create_feature_extractor(
        self, mel_spec_paramms, exportable, top_db, quantizable, spec_extractor
    ):
        if spec_extractor == "Melspec":
            if exportable:
                spec_init = TraceableMelspec
            else:
                spec_init = MelSpectrogram
        elif spec_extractor == "CQT":
            spec_init = CQT1992v2
        elif spec_extractor == "LEAF":
            spec_init = frontend.Leaf
        else:
            raise NotImplementedError(f"{spec_extractor} not implemented")
        if isinstance(mel_spec_paramms, list):
            self._n_specs = len(mel_spec_paramms)
            return nn.ModuleList(
                [
                    nn.Sequential(
                        spec_init(**mel_spec_paramm, quantizable=True)
                        if quantizable
                        else spec_init(**mel_spec_paramm),
                        QuantizableAmplitudeToDB(top_db=top_db)
                        if quantizable
                        else AmplitudeToDB(top_db=top_db),
                        NormalizeMelSpec(exportable=exportable),
                    )
                    for mel_spec_paramm in mel_spec_paramms
                ]
            )
        else:
            self._n_specs = 1
            if spec_extractor != "LEAF":
                return nn.Sequential(
                    spec_init(**mel_spec_paramms, quantizable=True)
                    if quantizable
                    else spec_init(**mel_spec_paramms),
                    QuantizableAmplitudeToDB(top_db=top_db)
                    if quantizable
                    else AmplitudeToDB(top_db=top_db),
                    NormalizeMelSpec(exportable=exportable),
                )
            else:
                if mel_spec_paramms.pop("normalize", False):
                    return nn.Sequential(
                        spec_init(**mel_spec_paramms, onnx_export=exportable),
                        NormalizeMelSpec(exportable=exportable),
                    )
                elif mel_spec_paramms.pop("normalize_and_db", False):
                    return nn.Sequential(
                        spec_init(**mel_spec_paramms, onnx_export=exportable),
                        AmplitudeToDB(top_db=top_db),
                        NormalizeMelSpec(exportable=exportable),
                    )
                else:
                    return spec_init(
                        **mel_spec_paramms, onnx_export=exportable
                    )

    def forward(self, input, return_spec_feature=False, return_cnn_emb=False):
        if self.central_crop_input is not None:
            overall_pad = input.shape[-1] // 2
            input = input[:, overall_pad // 2 : -(overall_pad // 2)]
        if self._n_specs > 1:
            spec = [
                mel_spec_extractor(input)[:, None]
                for mel_spec_extractor in self.logmelspec_extractor
            ]
            spec = torch.cat(spec, dim=1)
        else:
            spec = self.logmelspec_extractor(input)[:, None]
        if self.spec_augment is not None and self.training:
            spec = self.spec_augment(spec)
        if not self.quantizable and return_spec_feature:
            return spec
        if self.deep_supervision_steps is not None:
            emb = self.backbone(spec)[-self.deep_supervision_steps :]
        elif self.transformer_backbone:
            emb = self.backbone(spec)
        else:
            emb = self.backbone(spec)[-1]
        if self.permute_backbone_emb is not None:
            emb = emb.permute(*self.permute_backbone_emb)
        if not self.quantizable and return_cnn_emb:
            return emb
        if self.head is not None:
            logits = self.head(emb)
            if self.use_selected_indices:
                logits = logits[:, self.selected_indices]
            if self.sigmoid is not None:
                logits = self.sigmoid(logits)
            return logits
        else:
            if self.no_head_inference_mode:
                return self.sigmoid(emb)
            else:
                return {
                    "clipwise_logits_long": emb,
                    "clipwise_pred_long": self.sigmoid(emb),
                }


class WaveTDNNClasifier(WaveCNNAttenClasifier):
    def __init__(
        self,
        device: str,
        mel_spec_paramms: Dict[str, Any],
        tdnn_paramms: Dict[str, Any],
        top_db: float = 80.0,
        exportable: bool = False,
        central_crop_input: Optional[float] = None,
        selected_indices: Optional[List[int]] = None,
        output_type: Optional[str] = None,
    ):
        super().__init__(
            backbone=None,
            device=device,
            mel_spec_paramms=mel_spec_paramms,
            head_config=None,
            top_db=top_db,
            pretrained=False,
            first_conv_stride_overwrite=None,
            exportable=exportable,
            central_crop_input=central_crop_input,
            selected_indices=selected_indices,
        )
        if self._n_specs > 1:
            raise NotImplementedError(
                "TDNN is not implemented for multiple spectrogram"
            )
        self.tdnn = ECAPA_TDNN(
            input_size=mel_spec_paramms["n_mels"], **tdnn_paramms
        )
        self.output_type = output_type
        self.sigmoid = nn.Sigmoid()
        self.to(self.device)

    def forward(self, input, return_spec_feature=False):
        if self.central_crop_input is not None:
            overall_pad = input.shape[-1] // 2
            input = input[:, overall_pad // 2 : -(overall_pad // 2)]
        spec = self.logmelspec_extractor(input)
        if return_spec_feature:
            return spec
        logits = self.tdnn(spec.transpose(1, 2)).squeeze(-2)
        probs = self.sigmoid(logits)
        # Just for compatibility with WaveCNNAttenClasifier
        if self.output_type is None:
            return {
                "clipwise_logits_long": logits,
                "clipwise_pred_long": probs,
            }
        elif self.output_type == "clipwise_logits_long":
            return logits
        elif self.output_type == "clipwise_pred_long":
            return probs
