import collections
from copy import deepcopy
from itertools import combinations
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
from hyperopt import fmin, hp, tpe
from joblib import Parallel, delayed
from scipy.special import expit
from sklearn.metrics import roc_auc_score


def coobminations_with_moving_r(iterable, min_r, max_r):
    accum = []
    for r in range(min_r, max_r + 1):
        accum.extend(list(combinations(iterable, r)))
    return accum


def compute_weighted_blend_with_dist_thresh(arrays, weights, thresh):
    temp_pred = arrays[0].values.copy()
    temp_pred[:] = 0
    temp_denum = arrays[0].values.copy()
    temp_denum[:] = 0
    len_preds = len(arrays)
    for i in range(len_preds):
        cur_pred = arrays[i].values.copy()
        pos_mask = expit(arrays[i].values) > thresh
        neg_mask = expit(arrays[i].values) <= thresh

        cur_pred[neg_mask] = cur_pred[neg_mask] * weights[i]
        cur_pred[pos_mask] = cur_pred[pos_mask] * weights[len_preds + i]
        temp_pred = temp_pred + cur_pred

        temp_denum[neg_mask] = temp_denum[neg_mask] + weights[i]
        temp_denum[pos_mask] = temp_denum[pos_mask] + weights[len_preds + i]

    return temp_pred / temp_denum


def sort_dict_by_keys(input):
    return collections.OrderedDict(
        sorted({int(k_[1:]): v_ for k_, v_ in input.items()}.items())
    )


class HyperOptSWA:
    def __init__(
        self,
        search_preds,
        target,
        fold_ids,
        opt_steps: int = 300,
        swa_steps: int = 9,
        hp_dist: Callable = lambda len_sub_cols: [
            hp.uniform(f"s{i+1}", 0, 1) for i in range(len_sub_cols)
        ],
        use_fold_av_opt: bool = False,
        dist_threshold: Optional[float] = None,
    ):
        self.search_preds = search_preds
        self.target = target
        self.fold_ids = fold_ids

        self.local_df = pd.DataFrame(
            {"target": deepcopy(target), "folds": deepcopy(fold_ids)}
        )

        self.opt_steps = opt_steps
        self.swa_steps = swa_steps
        self.hp_dist = hp_dist
        self.use_fold_av_opt = use_fold_av_opt
        self.dist_threshold = dist_threshold

    def swa_opt_func(self, args):
        if self.dist_threshold is None:
            temp_pred = sum(
                el * coef for el, coef in zip(self.search_preds, args)
            )
            temp_pred /= sum(args)
        else:
            temp_pred = compute_weighted_blend_with_dist_thresh(
                self.search_preds, args, self.dist_threshold
            )

        temp_score = roc_auc_score(self.target, temp_pred)
        return -temp_score

    def swa_opt_func_fold_average(self, args):
        if self.dist_threshold is None:
            temp_pred = sum(
                el * coef for el, coef in zip(self.search_preds, args)
            )
            temp_pred /= sum(args)
        else:
            temp_pred = compute_weighted_blend_with_dist_thresh(
                self.search_preds, args, self.dist_threshold
            )
        self.local_df["pred"] = temp_pred
        temp_score = (
            self.local_df.groupby("folds")
            .apply(lambda x: roc_auc_score(x["target"], x["pred"]))
            .mean()
        )
        return -temp_score

    def swa_opt(self):
        best_coefs = [
            fmin(
                self.swa_opt_func_fold_average
                if self.use_fold_av_opt
                else self.swa_opt_func,
                self.hp_dist(
                    len(self.search_preds)
                    if self.dist_threshold is None
                    else 2 * len(self.search_preds)
                ),
                algo=tpe.suggest,
                max_evals=self.opt_steps,
            )
            for _ in range(self.swa_steps)
        ]
        best_coefs = np.array(
            [
                np.array(list(sort_dict_by_keys(el).values()))
                for el in best_coefs
            ]
        )
        best_swa_coefs = np.mean(best_coefs, axis=0)
        best_score = (
            self.swa_opt_func_fold_average(best_swa_coefs)
            if self.use_fold_av_opt
            else self.swa_opt_func(best_swa_coefs)
        )
        best_score = -best_score
        return best_score, best_coefs, best_swa_coefs

    def swa_opt_no_exception(self):
        try:
            return self.swa_opt()
        except Exception as e:
            print(f"Exception : {e}")
            return -1, [], []


def call_HyperOptSWA(HyperOptSWA_obj):
    return HyperOptSWA_obj.swa_opt()


def call_HyperOptSWA_noexception(HyperOptSWA_obj):
    return HyperOptSWA_obj.swa_opt_no_exception()


class ForwardBlend:
    def __init__(
        self,
        val_test_pair,
        ids,
        targets,
        fold_ids,
        grid_r_min: int = None,
        grid_r_max: int = None,
        opt_steps: int = 300,
        swa_steps: int = 9,
        patience: float = 1e-5,
        verbose: bool = True,
        hp_dist: Callable = lambda len_sub_cols: [
            hp.uniform(f"s{i+1}", 0, 1) for i in range(len_sub_cols)
        ],
        n_cores: int = 15,
        use_fold_av_opt: bool = False,
        apply_func: Optional[Callable] = None,
        use_backward_path: bool = False,
        start_sub: Optional[str] = None,
        dist_threshold: Optional[float] = None,
        supress_swa_exception: bool = False,
        selected_folds: Optional[List[int]] = None,
        omit_fold_selection: Optional[List[int]] = None,
    ):
        if apply_func is not None:
            self.print_v("Using apply function")
            self.val_dfs = {
                f"oof_{i}": apply_func(el[0])
                for i, el in enumerate(val_test_pair)
            }
            self.test_dfs = {
                f"test_{i}": apply_func(el[1])
                for i, el in enumerate(val_test_pair)
            }
        else:
            self.val_dfs = {
                f"oof_{i}": el[0] for i, el in enumerate(val_test_pair)
            }
            self.test_dfs = {
                f"test_{i}": el[1] for i, el in enumerate(val_test_pair)
            }

        if selected_folds is not None:
            mask = fold_ids.isin(selected_folds)
            for i in range(len(val_test_pair)):
                if i not in omit_fold_selection:
                    self.val_dfs[f"oof_{i}"] = self.val_dfs[f"oof_{i}"][mask]
                else:
                    print(f"oof_{i} does not fit mask")
            fold_ids = fold_ids[mask]
            targets = targets[mask]
            ids = ids[mask]

        self.ids = ids
        self.targets = targets
        self.fold_ids = fold_ids

        self.opt_steps = opt_steps
        self.swa_steps = swa_steps
        self.patience = patience
        self.verbose = verbose
        self.hp_dist = hp_dist
        self.n_cores = n_cores
        self.use_fold_av_opt = use_fold_av_opt
        self.use_backward_path = use_backward_path
        self.start_sub = start_sub
        self.dist_threshold = dist_threshold
        self.supress_swa_exception = supress_swa_exception

        self.grid_r_min = grid_r_min
        self.grid_r_max = grid_r_max

        self.history = []
        self.best_score = None
        self.best_coefs = None
        self.best_swa_coefs = None
        self.best_sub_coomb = None

        self.temp_swa_subs = None

    def print_v(self, msg):
        if self.verbose:
            print(msg)

    def update_stats_and_print(
        self,
        best_score,
        best_sub_coomb,
        best_coefs,
        best_swa_coefs,
        omit_history_update=False,
    ):
        self.best_score = best_score
        self.best_sub_coomb = best_sub_coomb
        self.best_coefs = best_coefs
        self.best_swa_coefs = best_swa_coefs
        if not omit_history_update:
            self.history.append(
                {
                    "best_score": best_score,
                    "best_sub_coomb": best_sub_coomb,
                    "best_swa_coefs": best_swa_coefs,
                    "best_coefs": best_coefs,
                }
            )
        self.print_v(f"best score : {best_score}")
        self.print_v(f"best sub coomb : {best_sub_coomb}")
        self.print_v(f"best swa coefs : {best_swa_coefs}")

    def oof2test_name(self, input):
        return "test_" + input.split("_")[1]

    def first_call(self):
        best_score = -np.inf
        best_sub_name = None
        if self.start_sub is None:
            for i in range(len(self.val_dfs)):
                if self.use_fold_av_opt:
                    temp_df = pd.DataFrame(
                        {
                            "target": deepcopy(self.targets),
                            "folds": deepcopy(self.fold_ids),
                            "pred": deepcopy(self.val_dfs[f"oof_{i}"]),
                        }
                    )
                    cur_score = (
                        temp_df.groupby("folds")
                        .apply(lambda x: roc_auc_score(x["target"], x["pred"]))
                        .mean()
                    )
                else:
                    cur_score = roc_auc_score(
                        self.targets, self.val_dfs[f"oof_{i}"]
                    )
                if cur_score > best_score:
                    best_score = cur_score
                    best_sub_name = f"oof_{i}"
        else:
            if self.use_fold_av_opt:
                temp_df = pd.DataFrame(
                    {
                        "target": deepcopy(self.targets),
                        "folds": deepcopy(self.fold_ids),
                        "pred": deepcopy(self.val_dfs[self.start_sub]),
                    }
                )
                best_score = (
                    temp_df.groupby("folds")
                    .apply(lambda x: roc_auc_score(x["target"], x["pred"]))
                    .mean()
                )
            else:
                best_score = roc_auc_score(
                    self.targets, self.val_dfs[f"oof_{i}"]
                )
            best_sub_name = self.start_sub

        return best_score, best_sub_name

    def not_first_call(self):
        opt_objects = []
        all_sub_coombs = []
        for i in range(len(self.val_dfs)):
            sub_name = f"oof_{i}"
            if sub_name not in self.best_sub_coomb:
                temp_swa_subs = deepcopy(self.best_sub_coomb) + [sub_name]
                opt_objects.append(
                    HyperOptSWA(
                        search_preds=[
                            self.val_dfs[el] for el in temp_swa_subs
                        ],
                        target=self.targets,
                        fold_ids=self.fold_ids,
                        opt_steps=self.opt_steps,
                        swa_steps=self.swa_steps,
                        hp_dist=self.hp_dist,
                        use_fold_av_opt=self.use_fold_av_opt,
                        dist_threshold=self.dist_threshold,
                    )
                )
                all_sub_coombs.append(temp_swa_subs)
        self.print_v("Started Parallel")
        if self.supress_swa_exception:
            temp_r = Parallel(n_jobs=self.n_cores)(
                delayed(call_HyperOptSWA_noexception)(hos_obj)
                for hos_obj in opt_objects
            )
        else:
            temp_r = Parallel(n_jobs=self.n_cores)(
                delayed(call_HyperOptSWA)(hos_obj) for hos_obj in opt_objects
            )
        best_scores = [el[0] for el in temp_r]
        self.print_v(f"Search best scores: {best_scores}")
        best_score_id = np.argmax(best_scores)
        best_score, best_coefs, best_swa_coefs = temp_r[best_score_id]
        best_sub_names = all_sub_coombs[best_score_id]
        return best_score, best_sub_names, best_coefs, best_swa_coefs

    def step_back(self, cur_sub):
        opt_objects = []
        all_sub_coombs = []
        best_sub_coomb_without_cur = deepcopy(self.best_sub_coomb)
        best_sub_coomb_without_cur.remove(cur_sub)
        for i in range(len(self.val_dfs)):
            sub_name = f"oof_{i}"
            if sub_name not in best_sub_coomb_without_cur:
                temp_swa_subs = deepcopy(best_sub_coomb_without_cur) + [
                    sub_name
                ]
                opt_objects.append(
                    HyperOptSWA(
                        search_preds=[
                            self.val_dfs[el] for el in temp_swa_subs
                        ],
                        target=self.targets,
                        fold_ids=self.fold_ids,
                        opt_steps=self.opt_steps,
                        swa_steps=self.swa_steps,
                        hp_dist=self.hp_dist,
                        use_fold_av_opt=self.use_fold_av_opt,
                        dist_threshold=self.dist_threshold,
                    )
                )
                all_sub_coombs.append(temp_swa_subs)
        self.print_v("Started Parallel")
        if self.supress_swa_exception:
            temp_r = Parallel(n_jobs=self.n_cores)(
                delayed(call_HyperOptSWA_noexception)(hos_obj)
                for hos_obj in opt_objects
            )
        else:
            temp_r = Parallel(n_jobs=self.n_cores)(
                delayed(call_HyperOptSWA)(hos_obj) for hos_obj in opt_objects
            )
        best_scores = [el[0] for el in temp_r]
        self.print_v(f"Search best scores: {best_scores}")
        best_score_id = np.argmax(best_scores)
        best_score, best_coefs, best_swa_coefs = temp_r[best_score_id]
        best_sub_names = all_sub_coombs[best_score_id]
        return best_score, best_sub_names, best_coefs, best_swa_coefs

    def fit(self):

        for step in range(len(self.val_dfs)):
            self.print_v(f"Itter {step} started")
            if step == 0:
                best_score, best_sub_name = self.first_call()
                self.update_stats_and_print(
                    best_score,
                    [best_sub_name],
                    best_coefs=np.array([1.0] * self.swa_steps),
                    best_swa_coefs=np.array([1.0]),
                )
            else:
                (
                    best_score,
                    best_sub_names,
                    best_coefs,
                    best_swa_coefs,
                ) = self.not_first_call()
                score_improvment = best_score - self.best_score
                self.print_v(f"Score improved: {score_improvment}")
                if score_improvment > self.patience:
                    self.update_stats_and_print(
                        best_score,
                        best_sub_names,
                        best_coefs=best_coefs,
                        best_swa_coefs=best_swa_coefs,
                    )
                else:
                    self.print_v("Stopped by patience")
                    if self.use_backward_path:
                        self.print_v("Starting backward path")
                        for step_back in reversed(
                            range(len(self.best_sub_coomb) - 1)
                        ):
                            self.print_v(f"Itter back {step_back} started")
                            self.print_v(
                                f"Distilling sub {self.best_sub_coomb[step_back]}"
                            )
                            (
                                best_score,
                                best_sub_names,
                                best_coefs,
                                best_swa_coefs,
                            ) = self.step_back(self.best_sub_coomb[step_back])
                            score_improvment = best_score - self.best_score
                            self.print_v(f"Score improved: {score_improvment}")
                            if score_improvment > self.patience:
                                self.update_stats_and_print(
                                    best_score,
                                    best_sub_names,
                                    best_coefs=best_coefs,
                                    best_swa_coefs=best_swa_coefs,
                                )
                    break
        self.print_v("Search ended")
        return self

    def fit_grid(self):
        all_keys = list(self.val_dfs.keys())
        self.print_v(f"Number of preds: {len(all_keys)}")
        all_key_combs = coobminations_with_moving_r(
            all_keys, self.grid_r_min, self.grid_r_max
        )
        all_key_combs = [list(el) for el in all_key_combs]
        self.print_v(f"Number of coombs: {len(all_key_combs)}")
        opt_objects = [
            HyperOptSWA(
                search_preds=[self.val_dfs[sub_el] for sub_el in el],
                target=self.targets,
                fold_ids=self.fold_ids,
                opt_steps=self.opt_steps,
                swa_steps=self.swa_steps,
                hp_dist=self.hp_dist,
                use_fold_av_opt=self.use_fold_av_opt,
                dist_threshold=self.dist_threshold,
            )
            for el in all_key_combs
        ]
        self.print_v(f"Number of opt_objects: {len(opt_objects)}")
        self.print_v("Started Parallel")
        if self.supress_swa_exception:
            self.history = Parallel(n_jobs=self.n_cores)(
                delayed(call_HyperOptSWA_noexception)(hos_obj)
                for hos_obj in opt_objects
            )
        else:
            self.history = Parallel(n_jobs=self.n_cores)(
                delayed(call_HyperOptSWA)(hos_obj) for hos_obj in opt_objects
            )
        best_scores = [el[0] for el in self.history]
        best_score_id = np.argmax(best_scores)
        best_score, best_coefs, best_swa_coefs = self.history[best_score_id]
        best_sub_names = all_key_combs[best_score_id]
        self.update_stats_and_print(
            best_score,
            best_sub_names,
            best_coefs=best_coefs,
            best_swa_coefs=best_swa_coefs,
        )
        self.print_v("Grid Search ended")

    def transform(self):
        if self.dist_threshold is None:
            best_oof = sum(
                self.val_dfs[el] * coef
                for el, coef in zip(self.best_sub_coomb, self.best_swa_coefs)
            )
            best_oof /= sum(self.best_swa_coefs)
            best_test = sum(
                self.test_dfs[self.oof2test_name(el)] * coef
                for el, coef in zip(self.best_sub_coomb, self.best_swa_coefs)
            )
            best_test /= sum(self.best_swa_coefs)
        else:
            best_oof = compute_weighted_blend_with_dist_thresh(
                [self.val_dfs[el] for el in self.best_sub_coomb],
                self.best_swa_coefs,
                self.dist_threshold,
            )
            best_test = compute_weighted_blend_with_dist_thresh(
                [
                    self.test_dfs[self.oof2test_name(el)]
                    for el in self.best_sub_coomb
                ],
                self.best_swa_coefs,
                self.dist_threshold,
            )
        return best_oof, best_test
