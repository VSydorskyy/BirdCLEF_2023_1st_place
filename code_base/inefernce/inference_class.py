import gc
import json
import math
import os
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..utils import groupby_np_array, load_json, stack_and_max_by_samples


def torch_cat_with_none(tensor_1, tensor_2):
    if tensor_1 is None:
        return tensor_2
    elif tensor_2 is None:
        return tensor_1
    else:
        return torch.cat([tensor_1, tensor_2])


def to_cpu_with_none(tensor):
    if tensor is None:
        return tensor
    else:
        return tensor.cpu()


class BirdsInference:
    def __init__(
        self,
        device,
        verbose=True,
        verbose_tqdm=True,
        use_long_short_preds=False,
        use_sigmoid=True,
        use_timewise_avarage=False,
        avarage_type="mean",
        model_output_key=None,
        use_sed_mode=False,
        sed_config=None,
    ):
        if use_sed_mode:
            if sed_config is None:
                raise ValueError("Define sed_config in order to use sed_mode")
            if use_long_short_preds:
                raise ValueError("SED does not support long_short model")

        self.verbose = verbose
        self.verbose_tqdm = verbose_tqdm
        self.device = device
        self.use_long_short_preds = use_long_short_preds
        self.use_sigmoid = use_sigmoid
        self.use_timewise_avarage = use_timewise_avarage
        self.use_sed_mode = use_sed_mode
        self.sed_config = sed_config
        assert avarage_type in ["mean", "vlom", "identity", "gaus"]
        self.avarage_type = avarage_type
        self.model_output_key = model_output_key

        self.val_pred = None
        if self.use_long_short_preds:
            print("Inference class in long short mode")
            self.val_pred_long = None
        self.val_tgt = None
        self.sample_ids = None

    def _print_v(self, msg):
        if self.verbose:
            print(msg)

    def _tqdm_v(self, generator):
        if self.verbose_tqdm:
            return tqdm(generator)
        else:
            return generator

    def _apply_pred_act(self, input):
        if self.use_sigmoid:
            input = torch.sigmoid(input)

        if self.use_timewise_avarage:
            input = input.max(axis=1)[0]

        return input

    def _avarage_preds(self, x):
        if self.avarage_type == "mean":
            return x.mean(0)
        elif self.avarage_type == "vlom":
            x1 = x.prod(axis=0) ** (1.0 / len(x))
            x = x**2
            x = x.mean(axis=0)
            x = x ** (1 / 2)
            return (x + x1) / 2
        elif self.avarage_type == "gaus":
            x = (x**2).mean(axis=0) ** 0.5
            return x
        else:
            return x

    def _model_forward(self, nn_model, wave):
        if self.use_long_short_preds:
            model_out = nn_model(wave.to(self.device))
            logits = (
                self._apply_pred_act(model_out["clipwise_pred_short"])
                .detach()
                .cpu()
            )
            logits_long = (
                self._apply_pred_act(model_out["clipwise_pred_long"])
                .detach()
                .cpu()
            )
            atten = None
        elif self.use_sed_mode:
            model_out = nn_model(wave.to(self.device))
            if self.sed_config["sed_avarage"] == "atten_w":
                atten = model_out["framewise_atten_short"].detach()
                logits = model_out["framewise_logits_short"].detach()
            else:
                atten = None
                logits = model_out["framewise_pred_short"].detach()
            logits_long = None
        else:
            logits = nn_model(wave.to(self.device))
            if self.model_output_key is not None:
                logits = logits[self.model_output_key]
            logits = self._apply_pred_act(logits).detach().cpu()
            logits_long = None
            atten = None

        return logits, logits_long, atten

    @torch.no_grad()
    def predict_val_loaders(
        self,
        nn_models,
        data_loaders,
    ):
        if isinstance(nn_models[0], list):
            ansamble = True
            n_folds = len(nn_models[0])
            n_models = len(nn_models)
            for i in range(n_models):
                for j in range(n_folds):
                    assert not nn_models[i][j].training
            nn_models = [
                [nn_models[j][i] for j in range(n_models)]
                for i in range(n_folds)
            ]
        else:
            ansamble = False
            for i in range(len(nn_models)):
                assert not nn_models[i].training

        all_preds, all_tgts = [], []
        if self.use_long_short_preds:
            all_preds_long = []

        for data_loader, nn_model in zip(data_loaders, nn_models):
            loader_preds, loader_tgts, loader_agg = [], [], []
            if self.use_long_short_preds:
                loader_preds_long = []
            if self.use_sed_mode:
                dfidx_pointer = None
                sed_ac = {
                    "logits": None,
                    "atten": None,
                    "end": None,
                    "start": None,
                    "dfidx": None,
                    "tgt": None,
                }

            for wave, target, dfidx, start, end in self._tqdm_v(data_loader):
                if self.use_sed_mode and dfidx_pointer is None:
                    dfidx_pointer = dfidx[0]
                if ansamble:
                    logits, logits_long, atten = [], [], []
                    for one_nn_model in nn_model:
                        logits_, logits_long_, atten_ = self._model_forward(
                            one_nn_model, wave
                        )
                        logits.append(logits_)
                        logits_long.append(logits_long_)
                        atten.append(atten_)
                    logits = self._avarage_preds(torch.stack(logits, axis=0))
                    if logits_long[0] is not None:
                        logits_long = self._avarage_preds(
                            torch.stack(logits_long, axis=0)
                        )
                    if self.use_sed_mode:
                        atten = self._avarage_preds(
                            torch.stack(logits_long, axis=0)
                        )
                else:
                    logits, logits_long, atten = self._model_forward(
                        nn_model, wave
                    )

                if self.use_sed_mode:
                    sed_ac["logits"] = torch_cat_with_none(
                        sed_ac["logits"], logits
                    )
                    if self.sed_config["sed_avarage"] == "atten_w":
                        sed_ac["atten"] = torch_cat_with_none(
                            sed_ac["atten"], atten
                        )
                    sed_ac["end"] = torch_cat_with_none(sed_ac["end"], end)
                    sed_ac["start"] = torch_cat_with_none(
                        sed_ac["start"], start
                    )
                    sed_ac["dfidx"] = torch_cat_with_none(
                        sed_ac["dfidx"], dfidx
                    )
                    sed_ac["tgt"] = torch_cat_with_none(sed_ac["tgt"], target)
                    el_id = 0
                    while el_id < len(sed_ac["dfidx"]):
                        one_dfidx = sed_ac["dfidx"][el_id]
                        if one_dfidx != dfidx_pointer:
                            (
                                processed_logits,
                                processed_dfidx,
                                _,
                            ) = self._compute_sed_on_one_sample(
                                current_dfiddx=dfidx_pointer,
                                test_end=sed_ac["end"][:el_id],
                                test_starts=sed_ac["start"][:el_id],
                                test_model_logits=sed_ac["logits"][:el_id],
                                test_model_attens=sed_ac["atten"][:el_id]
                                if self.sed_config["sed_avarage"] == "atten_w"
                                else None,
                            )
                            current_tgt = sed_ac["tgt"][0]

                            sed_ac["dfidx"] = sed_ac["dfidx"][el_id:]
                            sed_ac["end"] = sed_ac["end"][el_id:]
                            sed_ac["start"] = sed_ac["start"][el_id:]
                            sed_ac["logits"] = sed_ac["logits"][el_id:]
                            sed_ac["tgt"] = sed_ac["tgt"][el_id:]
                            if self.sed_config["sed_avarage"] == "atten_w":
                                sed_ac["atten"] = sed_ac["atten"][el_id:]
                            torch.cuda.empty_cache()
                            gc.collect()

                            loader_preds.append(processed_logits)
                            loader_agg.append(processed_dfidx)
                            loader_tgts.append(
                                current_tgt[None, ...].repeat(
                                    len(processed_dfidx), 1
                                )
                            )
                            dfidx_pointer = one_dfidx
                            el_id = 0
                        else:
                            el_id += 1
                else:
                    loader_preds.append(logits.cpu())
                    if self.use_long_short_preds:
                        loader_preds_long.append(logits_long.cpu())
                    loader_tgts.append(target.numpy())
                    loader_agg.append(dfidx.numpy())

            if self.use_sed_mode:
                (
                    processed_logits,
                    processed_dfidx,
                    _,
                ) = self._compute_sed_on_one_sample(
                    current_dfiddx=sed_ac["dfidx"][0],
                    test_end=sed_ac["end"],
                    test_starts=sed_ac["start"],
                    test_model_logits=sed_ac["logits"],
                    test_model_attens=sed_ac["atten"]
                    if self.sed_config["sed_avarage"] == "atten_w"
                    else None,
                )
                current_tgt = sed_ac["tgt"][0]

                del sed_ac
                torch.cuda.empty_cache()
                gc.collect()

                loader_preds.append(processed_logits)
                loader_agg.append(processed_dfidx)
                loader_tgts.append(
                    current_tgt[None, ...].repeat(len(processed_dfidx), 1)
                )

            loader_agg = np.concatenate(loader_agg)
            all_preds.append(
                groupby_np_array(
                    groupby_f=loader_agg,
                    array_to_group=np.concatenate(loader_preds),
                    apply_f=stack_and_max_by_samples,
                )
            )
            if self.use_long_short_preds:
                all_preds_long.append(
                    groupby_np_array(
                        groupby_f=loader_agg,
                        array_to_group=np.concatenate(loader_preds_long),
                        apply_f=stack_and_max_by_samples,
                    )
                )
            all_tgts.append(
                groupby_np_array(
                    groupby_f=loader_agg,
                    array_to_group=np.concatenate(loader_tgts),
                    apply_f=stack_and_max_by_samples,
                )
            )

        # all_preds = np.concatenate(all_preds)
        # all_tgts = np.concatenate(all_tgts)
        # if self.use_long_short_preds:
        #     all_preds_long = np.concatenate(all_preds_long)
        # else:
        #     all_preds_long = None

        if not self.use_long_short_preds:
            all_preds_long = None

        return all_tgts, all_preds, all_preds_long

    def _model_forward_test(
        self, nn_models, wave, to_cpu=True, is_onnx_model=False
    ):
        wave = wave.to(self.device)
        if is_onnx_model:
            models_logits = nn_models.run(
                None,
                {"input": wave.numpy()},
            )[0]
            models_logits_long = None
            model_attens = None
        elif self.use_long_short_preds:
            models_logits, models_logits_long = [], []
            for nn_model in nn_models:
                temp_m_out = nn_model(wave)
                models_logits.append(
                    temp_m_out["clipwise_pred_short"].detach()
                )
                models_logits_long.append(
                    temp_m_out["clipwise_pred_long"].detach()
                )
            models_logits = self._avarage_preds(
                torch.stack(self._apply_pred_act(models_logits), axis=0)
            )
            models_logits_long = self._avarage_preds(
                torch.stack(self._apply_pred_act(models_logits_long), axis=0)
            )
            model_attens = None
        elif self.use_sed_mode:
            models_logits = []
            if self.sed_config["sed_avarage"] == "atten_w":
                model_attens = []
            for nn_model in nn_models:
                temp_m_out = nn_model(wave)
                if self.sed_config["sed_avarage"] == "atten_w":
                    model_attens.append(
                        temp_m_out["framewise_atten_short"].detach()
                    )
                    models_logits.append(
                        temp_m_out["framewise_logits_short"].detach()
                    )
                else:
                    models_logits.append(
                        temp_m_out["framewise_pred_short"].detach()
                    )
            models_logits = self._avarage_preds(
                torch.stack(models_logits, axis=0)
            )
            if self.sed_config["sed_avarage"] == "atten_w":
                model_attens = self._avarage_preds(
                    torch.stack(model_attens, axis=0)
                )
            else:
                model_attens = None
            models_logits_long = None
        else:
            if self.model_output_key is None:
                models_logits = torch.stack(
                    [
                        self._apply_pred_act(nn_model(wave).detach())
                        for nn_model in nn_models
                    ],
                    axis=0,
                )
            else:
                models_logits = torch.stack(
                    [
                        self._apply_pred_act(
                            nn_model(wave)[self.model_output_key].detach()
                        )
                        for nn_model in nn_models
                    ],
                    axis=0,
                )
            models_logits = self._avarage_preds(models_logits)
            models_logits_long = None
            model_attens = None

        if to_cpu:
            return (
                to_cpu_with_none(models_logits),
                to_cpu_with_none(models_logits_long),
                to_cpu_with_none(model_attens),
            )
        else:
            return (models_logits, models_logits_long, model_attens)

    def _compute_sed_on_one_sample(
        self,
        current_dfiddx,
        test_end,
        test_starts,
        test_model_logits,
        test_model_attens=None,
    ):
        fm = self.sed_config["frame_multip"]
        ip = self.sed_config["infer_period"]
        sr = self.sed_config["sr"]

        total_sec = test_end.max()
        pred_cont = torch.zeros(
            (int(total_sec) * fm, test_model_logits.shape[-1])
        ).to(test_model_logits.device)
        if test_model_attens is not None:
            atten_cont = torch.zeros(
                (int(total_sec) * fm, test_model_logits.shape[-1])
            ).to(test_model_logits.device)
        n_samples = torch.zeros((int(total_sec) * fm, 1)).to(
            test_model_logits.device
        )
        for sample_idx, (sec_start, framewise_pred) in enumerate(
            # `+ ip` for ignoring lookback
            zip(
                (test_starts / sr + ip).long(),
                test_model_logits,
            )
        ):
            sample_len = min(pred_cont.shape[0] - sec_start * fm, ip * fm)

            pred_cont[
                sec_start * fm : sec_start * fm + sample_len
            ] += framewise_pred[:sample_len]
            if test_model_attens is not None:
                atten_cont[
                    sec_start * fm : sec_start * fm + sample_len
                ] += test_model_attens[sample_idx, :sample_len]
            n_samples[sec_start * fm : sec_start * fm + sample_len] += 1

        pred_cont = pred_cont / n_samples
        if test_model_attens is not None:
            atten_cont = atten_cont / n_samples

        if self.sed_config["sed_avarage"] == "max":
            assert 0 <= pred_cont.max() <= 1
            assert 0 <= pred_cont.min() <= 1

        pred_cont = [
            pred_cont[i * ip * fm : (i + 1) * ip * fm]
            for i in range(math.ceil(pred_cont.shape[0] / (ip * fm)))
        ]
        if pred_cont[-1].shape[0] != ip * fm:
            pad_tuple = (0, 0, 0, ip * fm - pred_cont[-1].shape[0])
            pred_cont[-1] = F.pad(
                pred_cont[-1],
                pad_tuple,
                "constant",
                self.sed_config["pred_pad"],
            )
        pred_cont = torch.stack(pred_cont)
        if test_model_attens is not None:
            atten_cont = [
                atten_cont[i * ip * fm : (i + 1) * ip * fm]
                for i in range(math.ceil(atten_cont.shape[0] / (ip * fm)))
            ]
            if atten_cont[-1].shape[0] != ip * fm:
                atten_cont = (0, 0, 0, ip * fm - atten_cont[-1].shape[0])
                atten_cont[-1] = F.pad(
                    atten_cont[-1],
                    pad_tuple,
                    "constant",
                    self.sed_config["atten_pad"],
                )
            atten_cont = torch.stack(atten_cont)

        if self.sed_config["sed_avarage"] == "atten_w":
            pred_cont = torch.sum(
                torch.sigmoid(pred_cont) * torch.softmax(atten_cont, axis=1),
                axis=1,
            )
        elif self.sed_config["sed_avarage"] == "max":
            pred_cont = torch.max(pred_cont, axis=1)[0]
        else:
            raise ValueError(
                f"{self.sed_config['sed_avarage']} is invalid SED avarage type"
            )
        assert 0 <= pred_cont.max() <= 1
        assert 0 <= pred_cont.min() <= 1

        return (
            pred_cont.cpu(),
            np.full(len(pred_cont), current_dfiddx),
            np.array([(i + 1) * ip for i in range(len(pred_cont))]).astype(
                int
            ),
        )

    @torch.no_grad()
    def predict_test_loader(self, nn_models, data_loader, is_onnx_model=False):
        if not is_onnx_model:
            if isinstance(nn_models[0], list):
                ansamble = True
                print("Ansambling")
                if self.use_sed_mode:
                    raise ValueError("Sed mode is not supported in ansambling")
            else:
                ansamble = False

            if ansamble:
                n_folds = len(nn_models[0])
                n_models = len(nn_models)
                for i in range(n_models):
                    for j in range(n_folds):
                        assert not nn_models[i][j].training
            else:
                for i in range(len(nn_models)):
                    assert not nn_models[i].training
        else:
            ansamble = False

        test_model_logits = []
        test_dfidx = []
        test_end = []
        if self.use_long_short_preds:
            test_model_logits_long = []
        if self.use_sed_mode:
            dfidx_pointer = None
            sed_ac = {
                "logits": None,
                "atten": None,
                "end": None,
                "start": None,
                "dfidx": None,
            }

        for wave, target, dfidx, start, end in self._tqdm_v(data_loader):
            if self.use_sed_mode and dfidx_pointer is None:
                dfidx_pointer = dfidx[0]
            if ansamble:
                models_logits, models_logits_long, model_atten = [], [], []
                for nn_model_exp in nn_models:
                    (
                        models_logits_,
                        models_logits_long_,
                        model_atten_,
                    ) = self._model_forward_test(
                        nn_models=nn_model_exp, wave=wave
                    )
                    models_logits.append(models_logits_)
                    models_logits_long.append(models_logits_long_)
                    model_atten.append(model_atten_)
                models_logits = self._avarage_preds(
                    torch.stack(models_logits, axis=0)
                ).numpy()
                if self.use_long_short_preds:
                    models_logits_long = self._avarage_preds(
                        torch.stack(models_logits_long, axis=0)
                    ).numpy()
                if self.use_sed_mode:
                    model_atten = self._avarage_preds(
                        torch.stack(model_atten, axis=0)
                    ).numpy()
            else:
                (
                    models_logits,
                    models_logits_long,
                    model_atten,
                ) = self._model_forward_test(
                    nn_models=nn_models,
                    wave=wave,
                    to_cpu=not self.use_sed_mode and not is_onnx_model,
                    is_onnx_model=is_onnx_model,
                )

            if self.use_sed_mode:
                sed_ac["logits"] = torch_cat_with_none(
                    sed_ac["logits"], models_logits
                )
                if self.sed_config["sed_avarage"] == "atten_w":
                    sed_ac["atten"] = torch_cat_with_none(
                        sed_ac["atten"], model_atten
                    )
                sed_ac["end"] = torch_cat_with_none(sed_ac["end"], end)
                sed_ac["start"] = torch_cat_with_none(sed_ac["start"], start)
                sed_ac["dfidx"] = torch_cat_with_none(sed_ac["dfidx"], dfidx)
                el_id = 0
                while el_id < len(sed_ac["dfidx"]):
                    one_dfidx = sed_ac["dfidx"][el_id]
                    if one_dfidx != dfidx_pointer:
                        print(f"Processing {dfidx_pointer} sample")
                        (
                            processed_logits,
                            processed_dfidx,
                            processed_ends,
                        ) = self._compute_sed_on_one_sample(
                            current_dfiddx=dfidx_pointer,
                            test_end=sed_ac["end"][:el_id],
                            test_starts=sed_ac["start"][:el_id],
                            test_model_logits=sed_ac["logits"][:el_id],
                            test_model_attens=sed_ac["atten"][:el_id]
                            if self.sed_config["sed_avarage"] == "atten_w"
                            else None,
                        )
                        sed_ac["dfidx"] = sed_ac["dfidx"][el_id:]
                        sed_ac["end"] = sed_ac["end"][el_id:]
                        sed_ac["start"] = sed_ac["start"][el_id:]
                        sed_ac["logits"] = sed_ac["logits"][el_id:]
                        if self.sed_config["sed_avarage"] == "atten_w":
                            sed_ac["atten"] = sed_ac["atten"][el_id:]
                        torch.cuda.empty_cache()
                        gc.collect()

                        test_model_logits.append(processed_logits)
                        test_dfidx.append(processed_dfidx)
                        test_end.append(processed_ends)
                        dfidx_pointer = one_dfidx
                        el_id = 0
                    else:
                        el_id += 1
            else:
                test_model_logits.append(models_logits)
                if self.use_long_short_preds:
                    test_model_logits_long.append(models_logits_long)
                test_dfidx.append(dfidx.numpy())
                test_end.append(end.numpy())

        if self.use_sed_mode:
            print(f"Processing {dfidx_pointer} sample")
            (
                processed_logits,
                processed_dfidx,
                processed_ends,
            ) = self._compute_sed_on_one_sample(
                current_dfiddx=sed_ac["dfidx"][0],
                test_end=sed_ac["end"],
                test_starts=sed_ac["start"],
                test_model_logits=sed_ac["logits"],
                test_model_attens=sed_ac["atten"]
                if self.sed_config["sed_avarage"] == "atten_w"
                else None,
            )
            del sed_ac
            torch.cuda.empty_cache()
            gc.collect()

            test_model_logits.append(processed_logits)
            test_dfidx.append(processed_dfidx)
            test_end.append(processed_ends)

        test_model_logits = np.concatenate(test_model_logits)
        if self.use_long_short_preds:
            test_model_logits_long = np.concatenate(test_model_logits_long)
        else:
            test_model_logits_long = None
        test_dfidx = np.concatenate(test_dfidx)
        test_end = np.concatenate(test_end)

        assert len(test_model_logits.shape) == 2

        return test_model_logits, test_model_logits_long, test_dfidx, test_end
