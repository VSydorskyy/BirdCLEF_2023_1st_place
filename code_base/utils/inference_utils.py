import os
from collections import OrderedDict
from copy import deepcopy
from os.path import join as pjoin
from typing import Any, Callable, List, Mapping, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def get_mode_model(model, mode="train"):
    if isinstance(model, dict):
        return model[mode]
    return model


def apply_avarage_weights_on_swa_path(
    swa_path: str,
    use_distributed: bool = False,
    take_best: Optional[int] = None,
    ascending: bool = False,
    verbose: bool = True,
):
    chkp = torch.load(
        swa_path,
        map_location="cpu",
    )
    chkp = sorted(
        [v for k, v in chkp.items()],
        key=lambda x: x[0] if ascending else -x[0],
    )
    if verbose:
        print(" ; ".join([f"Itter {v[2]} Score {v[0]}" for v in chkp]))
    return avarage_weights(
        [v[1] for v in chkp],
        delete_module=use_distributed,
        take_best=take_best,
    )


def avarage_weights(
    nn_weights: List[OrderedDict],
    delete_module: bool = False,
    take_best: Optional[int] = None,
):
    if take_best is not None:
        print("solo model")
        avaraged_dict = OrderedDict()
        for k in nn_weights[take_best].keys():
            if delete_module:
                new_k = k[len("module.") :]
            else:
                new_k = k

            avaraged_dict[new_k] = nn_weights[take_best][k]
    else:
        n_nns = len(nn_weights)
        if n_nns < 2:
            raise RuntimeError("Please provide more then 2 checkpoints")

        avaraged_dict = OrderedDict()
        for k in nn_weights[0].keys():
            if delete_module:
                new_k = k[len("module.") :]
            else:
                new_k = k

            avaraged_dict[new_k] = sum(
                nn_weights[i][k] for i in range(n_nns)
            ) / float(n_nns)

    return avaraged_dict


def get_validation_models(
    model_initilizer: Callable,
    model_config: Mapping[str, Any],
    model_ckp_dicts: List[OrderedDict],
    device: str,
):
    t_models = []

    for mcd in model_ckp_dicts:

        try:
            t_model = model_initilizer(**model_config, device=device)
            t_model.load_state_dict(mcd)
        except:
            print("OLD STYLE MODEL")
            temp_config = deepcopy(model_config)
            temp_config["old_style_model"] = True
            t_model = model_initilizer(**temp_config, device=device)
            t_model.load_state_dict(mcd)

        t_model = t_model.to(device)
        t_model.eval()
        t_models.append(t_model)

    return t_models


def create_val_loaders(
    loader_initilizer: object,
    loader_config: Mapping[str, Any],
    dfs: List[str],
    batch_size: int,
):
    t_loaders = []

    for df in dfs:
        t_dataset = loader_initilizer(df=df, **loader_config)
        t_loader = torch.utils.data.DataLoader(
            t_dataset,
            batch_size=batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=8,
        )

        t_loaders.append(t_loader)

    return t_loaders


def predict_over_all_train(
    my_loaders,
    my_models,
    model_predict_func,
    device,
    do_concat=True,
):
    logits = []
    for loader, model in zip(my_loaders, my_models):
        for batch in tqdm(loader):
            logit = model_predict_func(batch, model, device)
            logits.append(logit)

    if do_concat:
        logits = np.concatenate(logits)

    return logits


def predict_test_with_multiple_models(
    my_models: List[torch.nn.Module],
    my_loader: torch.utils.data.DataLoader,
    predict_func: Callable,
    device: str,
    do_concat=True,
):
    logits = []
    for batch in tqdm(my_loader):
        logit = np.stack(
            [predict_func(batch, m, device) for m in my_models],
            axis=0,
        )
        logits.append(logit)

    if do_concat:
        logits = np.concatenate(logits, axis=1)

    return logits


@torch.no_grad()
def base_model_predict(t_batch, t_model, t_device):
    wave = t_batch[0].to(t_device)
    logits = t_model(wave)
    logits = logits[:, 0].detach().cpu().numpy()
    return logits


def compose_submission_dataframe(
    probs, dfidxs, end_seconds, filenames, bird2id
):
    id2bird = {v: k for k, v in bird2id.items()}
    test_pred = pd.DataFrame(
        columns=["row_id"] + [id2bird[i] for i in range(len(id2bird))]
    )

    filenames = filenames.apply(
        lambda x: os.path.splitext(os.path.basename(x))[0]
    )
    test_pred["row_id"] = [
        filenames.iloc[df_idx] + "_" + str(end_s)
        for df_idx, end_s in zip(dfidxs, end_seconds)
    ]

    test_pred.iloc[:, 1:] = probs

    return test_pred
