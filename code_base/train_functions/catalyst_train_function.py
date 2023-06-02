import copy
import json
import re
from collections import OrderedDict
from pprint import pprint
from typing import Callable, List, Optional, OrderedDict, Union

import numpy as np
import pandas as pd
import torch
import torch.quantization.quantize_fx as quantize_fx
from catalyst import dl, metrics
from torch.ao.quantization import QConfigMapping

from ..utils.inference_utils import (
    apply_avarage_weights_on_swa_path,
    get_mode_model,
)


class CustomRunner(dl.Runner):
    def _dynamic_meters_updated(self, batch_metrics_dict):
        if len(batch_metrics_dict) > len(self.meters.keys()):
            additional_loss_metric_names = list(
                set(batch_metrics_dict.keys()) - set(self.meters.keys())
            )
            for add_key in additional_loss_metric_names:
                self.meters[add_key] = metrics.AdditiveMetric(
                    compute_on_call=False
                )
        for key in batch_metrics_dict.keys():
            self.meters[key].update(
                self.batch_metrics[key].item(), self.batch_size
            )

    def on_loader_start(self, runner):
        super().on_loader_start(runner)
        self.meters = {}

    def on_loader_end(self, runner):
        for key in self.meters.keys():
            self.loader_metrics[key] = self.meters[key].compute()[0]
        super().on_loader_end(runner)

    def handle_batch(self, batch):

        losses, inputs, outputs = self.criterion(self, batch)

        self.batch_metrics.update(losses)
        self._dynamic_meters_updated(losses)
        self.input = inputs
        self.output = outputs

    def on_loader_end(self, runner):
        for key in self.meters.keys():
            self.loader_metrics[key] = self.meters[key].compute()[0]
        super().on_loader_end(runner)


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def filter_state_dict(state_dict, regex):
    filtered_state_dict = OrderedDict()
    pattern = re.compile(regex)

    for key, value in state_dict.items():
        if not pattern.match(key):
            filtered_state_dict[key] = value

    return filtered_state_dict


def load_checkpoint(
    checkpoint_path,
    input_model,
    checkpoint_type: str = "catalyst",
    use_distributed: bool = False,
    filter_regex: Optional[str] = None,
    strict: bool = True,
):
    if checkpoint_type == "catalyst":
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    elif checkpoint_type == "swa":
        checkpoint = apply_avarage_weights_on_swa_path(
            checkpoint_path, use_distributed=use_distributed
        )
    else:
        raise RuntimeError(
            f"{checkpoint_type} is invalid value for `checkpoint_type`"
        )
    if filter_regex is not None:
        checkpoint = filter_state_dict(checkpoint, filter_regex)
    strict = filter_regex is None and strict
    if not strict:
        chkp_keys = set(checkpoint.keys())
        model_keys = set(input_model.state_dict().keys())
        print(f"Missing keys in t_chkp: {model_keys - chkp_keys}")
        print(f"Missing keys in t_model: {chkp_keys - model_keys}")
    input_model.load_state_dict(checkpoint, strict=strict)
    print("Pretrain Model Loaded")
    return input_model


def omit_errors_in_dataset_create(
    dataset_class, df, dataset_config, n_times=3, use_sampler=False
):
    n_exceptions = 0
    while True:
        if n_exceptions < n_times - 1:
            try:
                return dataset_class(
                    df=df,
                    use_sampler=use_sampler,
                    **dataset_config,
                )
            except Exception as e:
                print(f"attempt {n_exceptions + 1} failed with {e}")
                n_exceptions += 1
        else:
            return dataset_class(
                df=df,
                use_sampler=use_sampler,
                **dataset_config,
            )


def catalyst_training(
    train_df: Optional[pd.DataFrame],
    val_df: Optional[pd.DataFrame],
    exp_name: str,
    seed: int,
    train_dataset_class: torch.utils.data.Dataset,
    val_dataset_class: Optional[torch.utils.data.Dataset],
    train_dataset_config: dict,
    val_dataset_config: Optional[dict],
    train_dataloader_config: dict,
    val_dataloader_config: Optional[dict],
    nn_model_class: torch.nn.Module,
    nn_model_config: dict,
    optimizer_init: Callable,
    scheduler_init: Callable,
    forward: Union[torch.nn.Module, Callable],
    n_epochs: int,
    catalyst_callbacks: Callable,
    main_metric: str,
    minimize_metric: bool,
    use_amp: bool = False,
    check: bool = False,
    pretrained_chekpoints: Optional[Union[List[str], str]] = None,
    checkpoint_type: str = "catalyst",
    use_one_checkpoint: bool = False,
    checkpoint_regex: Optional[str] = None,
    load_checkpoint_strict: bool = True,
    infer_dataset_class: Optional[torch.utils.data.Dataset] = None,
    infer_dataset_config: Optional[dict] = None,
    infer_dataloader_config: Optional[torch.utils.data.Dataset] = None,
    valid_loader: str = "valid",
    create_ema_model: bool = False,
    class_weights_path: Optional[str] = None,
    label_str2int_path: Optional[str] = None,
    selected_birds: Optional[List[str]] = None,
    use_sampler: bool = False,
    qat: bool = False,
    qat_get_example_input: Optional[Callable] = None,
    qat_backend: str = "qnnpack",
):
    if qat:
        if use_amp:
            raise ValueError("QAT and AMP are mutually exclusive")
        if create_ema_model:
            raise ValueError("QAT and EMA are mutually exclusive")
    # Set numpy reproducibility
    np.random.seed(seed)
    # Set device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Training Device : {device}")

    omit_val = (
        val_dataset_class is None
        or val_dataset_config is None
        or val_dataloader_config is None
    )
    if omit_val:
        print("Omitting VALIDATION!!!!!")

    use_class_weights = class_weights_path is not None
    if use_class_weights and label_str2int_path is None:
        raise ValueError("Class weights require label_str2int mapping")
    if use_sampler and not use_class_weights:
        raise ValueError("Sampler requires class weights")
    train_dataset = train_dataset_class(
        df=train_df,
        use_sampler=use_sampler,
        **train_dataset_config,
    )
    # omit_errors_in_dataset_create(
    #     dataset_class=train_dataset_class,
    #     df=train_df,
    #     dataset_config=train_dataset_config,
    #     use_sampler=use_class_weights,
    # )
    if use_class_weights:
        class_weights = json.load(open(class_weights_path))
        print("Using next class weights:")
        pprint(class_weights)
    if use_class_weights and use_sampler:
        sample_weights = np.array(
            [class_weights[el] for el in train_dataset.targets]
        )
        assert len(sample_weights) == len(train_dataset)
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights, len(sample_weights)
        )
        print("Sampler Created")
    else:
        sampler = None
    if not omit_val:
        val_dataset = val_dataset_class(
            df=val_df,
            use_sampler=use_sampler,
            **val_dataset_config,
        )
        # val_dataset = omit_errors_in_dataset_create(
        #     dataset_class=val_dataset_class,
        #     df=val_df,
        #     dataset_config=val_dataset_config,
        # )

    loaders = {
        "train": torch.utils.data.DataLoader(
            train_dataset,
            worker_init_fn=worker_init_fn,
            sampler=sampler,
            **train_dataloader_config,
        ),
    }
    if not omit_val:
        loaders["valid"] = torch.utils.data.DataLoader(
            val_dataset, worker_init_fn=worker_init_fn, **val_dataloader_config
        )

    if infer_dataset_class is not None:
        add_dataset = omit_errors_in_dataset_create(
            dataset_class=infer_dataset_class,
            df=val_df,
            dataset_config=infer_dataset_config,
        )
        loaders["infer"] = torch.utils.data.DataLoader(
            add_dataset,
            worker_init_fn=worker_init_fn,
            **infer_dataloader_config,
        )

    model = nn_model_class(device=device, **nn_model_config)

    if pretrained_chekpoints is not None:
        if use_one_checkpoint:
            print(f"Loading model: {pretrained_chekpoints}")
            model = load_checkpoint(
                pretrained_chekpoints,
                model,
                checkpoint_type,
                filter_regex=checkpoint_regex,
                strict=load_checkpoint_strict,
            )
        else:
            fold_id = int(exp_name.split("/")[-1].split("_")[-1])
            print(f"Loading model: {pretrained_chekpoints[fold_id]}")
            model = load_checkpoint(
                pretrained_chekpoints[fold_id],
                model,
                checkpoint_type,
                filter_regex=checkpoint_regex,
                strict=load_checkpoint_strict,
            )

    if create_ema_model:
        model = {"train": model}
        model["val"] = nn_model_class(device=device, **nn_model_config)
        model["val"].load_state_dict(model["train"].state_dict())

    print(get_mode_model(model))
    for k in loaders.keys():
        print(f"{k} Loader Len = {len(loaders[k])}")

    optimizer = optimizer_init(get_mode_model(model))
    scheduler = scheduler_init(optimizer, len(loaders["train"]))

    runner = CustomRunner()

    if not isinstance(forward, torch.nn.Module):
        forward = forward()

    if hasattr(forward, "use_weights") and forward.use_weights:
        print("Setting loss weights ...")
        label_str2int = json.load(open(label_str2int_path))
        class_weights_array = (
            pd.Series(
                {
                    label_str2int[k]: class_weights[k]
                    for k in label_str2int.keys()
                }
            )
            .sort_index()
            .values.astype(np.float32)
        )
        forward.set_weights(class_weights_array, device)

    if hasattr(forward, "use_slected_indices") and forward.use_slected_indices:
        print("Setting slected_indices ...")
        label_str2int = json.load(open(label_str2int_path))
        forward.set_selected_indices(
            [label_str2int[el] for el in selected_birds], device
        )

    if qat:
        torch.backends.quantized.engine = qat_backend
        print("Using QAT")
        example_inputs = qat_get_example_input()
        qconfig_mapping = (
            torch.ao.quantization.get_default_qat_qconfig_mapping(qat_backend)
        )
        qconfig_mapping.set_module_name_regex("logmelspec_extractor.*", None)
        qconfig_mapping.set_module_name_regex("head.*", None)
        model.train()
        model_prepared = quantize_fx.prepare_qat_fx(
            model, qconfig_mapping, example_inputs
        )
        print(model_prepared)

    runner.train(
        model=model_prepared if qat else model,
        optimizer=optimizer,
        criterion=forward,
        scheduler=scheduler,
        loaders=loaders,
        logdir=exp_name,
        num_epochs=n_epochs,
        seed=seed,
        verbose=True,
        load_best_on_end=False,
        valid_loader=valid_loader,
        valid_metric=main_metric,
        timeit=True,
        check=check,
        minimize_valid_metric=minimize_metric,
        fp16=use_amp,
        callbacks=catalyst_callbacks(),  # We need to call this to make unique objects
    )  # for each fold

    return runner
