"""
Utilities for working with files.
"""

import os
import pickle as pkl
from copy import deepcopy
from typing import Dict, Any, Callable, Optional, cast, List, Tuple, Iterator, Union

import torch
import numpy as np

from experiment_utils import configs, utils


def save_experiment(
    exp_dict: dict,
    results_dir: str = "results",
    results: Optional[Any] = None,
    metrics: Optional[Dict[str, Any]] = None,
    model: Optional[Any] = None,
):
    """Save results of the experiment corresponding to the given dictionary.
    :param exp_dict: experiment dictionary.
    :param results_dir: base directory for experimental results.
    :param results: the raw return value of the experiment.
    :param metrics: (optional) dictionary of metrics collected during the experiment.
    :param model: (optional) the optimized model to be serialized.
    """

    hash_id = configs.hash_dict(exp_dict)
    path = os.path.join(results_dir, hash_id)

    # make directory if it doesn't exist.
    os.makedirs(path, exist_ok=True)

    if results is not None:
        with open(os.path.join(path, "return_value.pkl"), "wb") as f:
            pkl.dump(results, f)

    if metrics is not None:
        with open(os.path.join(path, "metrics.pkl"), "wb") as f:
            pkl.dump(metrics, f)

    if model is not None:
        # save a PyTorch model using torch builtins.
        if isinstance(model, torch.nn.Module):
            torch_path = os.path.join(path, "model.pt")
            model = cast(torch.nn.Module, model)

            # extract the state dictionary and save the model.
            state_dict = model.state_dict()
            torch.save(state_dict, torch_path)
        else:
            # pickle the model as usual.
            pkl_path = os.path.join(path, "model.pkl")
            with open(pkl_path, "wb") as f:
                pkl.dump(model, f)

    return results


def load_experiment(
    exp_dict: dict = None,
    hash_id: str = None,
    results_dir: Union[List[str], str] = "results",
    load_metrics: bool = False,
    load_model: bool = False,
) -> Dict[str, Any]:
    """Load results of the experiment corresponding to the given dictionary.
    :param exp_dict: experiment dictionary.
    :param results_dir: base directory or list of base directories for experimental results.
    :param load_metrics: whether or not to load metrics from the experiment.
    :param load_model: whether or not to load a model associated with the experiment.
    :returns: dict containing results. It is indexed by 'return_value' and (optionally) 'metrics', 'model'.
    """

    if hash_id is None:
        if exp_dict is None:
            raise ValueError("One of 'exp_dict' or 'hash_id' must not be 'None'.")

        hash_id = configs.hash_dict(exp_dict)

    results = {}

    success = False
    for src in utils.as_list(results_dir):
        path = os.path.join(src, hash_id)
        if os.path.exists(path):
            success = True
            break

    if not success:
        raise ValueError(f"Cannot find experiment {exp_dict} in one of {results_dir}!")

    try:
        with open(os.path.join(path, "return_value.pkl"), "rb") as f:
            results["return_value"] = pkl.load(f)
    except FileNotFoundError:
        results["return_value"] = None

    if load_metrics:
        with open(os.path.join(path, "metrics.pkl"), "rb") as f:
            results["metrics"] = pkl.load(f)

    if load_model:
        pkl_path = os.path.join(path, "model.pkl")
        torch_path = os.path.join(path, "model.pt")

        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                results["model"] = pkl.load(f)
        elif os.path.exists(torch_path):
            results["model"] = torch.load(torch_path)

    return results


def load_metric_grid(
    grid: dict,
    results_dir: Union[List[str], str] = "results",
    processing_fn: Optional[Callable] = None,
) -> dict:
    """Load metrics according to a supplied grid of experiment dictionaries.
    :param grid: a grid of experiment dictionaries. See 'configs.make_grid'.
    :param results_dir: base directory or list of base directories for experimental results.
    :param processing_fn: (optional) a function to call on each experiment to process the results.
    :returns: nested dictionary with the same "shape" as 'grid' containing the loaded metrics.
    """

    results_grid = deepcopy(grid)
    if processing_fn is None:

        def processing_fn(vals, keys):
            return vals

    for row in grid.keys():
        for metric_name in grid[row].keys():
            for line in grid[row][metric_name].keys():
                for repeat in grid[row][metric_name][line].keys():
                    vals = load_experiment(
                        exp_dict=grid[row][metric_name][line][repeat],
                        results_dir=results_dir,
                        load_metrics=True,
                        load_model=False,
                    )["metrics"][metric_name]

                    results_grid[row][metric_name][line][repeat] = processing_fn(
                        np.array(vals),
                        (row, metric_name, line, repeat),
                    )

    return results_grid


def compute_metrics(
    metric_grid: dict,
    metric_fn: Optional[Callable] = None,
    x_key: str = None,
    x_vals: Union[List, np.ndarray] = None,
) -> dict:
    """Load metrics according to a supplied grid of experiment dictionaries.
    :param metric_grid: a grid of experiment dictionaries with loaded metrics.
    :param metric_fn: (optional) function to process metrics after they are loaded.
    :param x_key: (optional) metric to use as "x axis" information (e.g. "time").
    :param x_vals: (optional) x-axis values. Cannot be supplied at the same time as 'x_key'.
    :returns: nested dictionary with the same "shape" as 'grid' containing the loaded metrics.
    """

    # both cannot be provided.
    assert x_key is None or x_vals is None

    results_grid = deepcopy(metric_grid)

    if metric_fn is None:

        def metric_fn(x):
            return x

    for row in metric_grid.keys():
        for metric_name in metric_grid[row].keys():
            for line in metric_grid[row][metric_name].keys():
                results = []
                repeats = metric_grid[row][metric_name][line].keys()
                for repeat in repeats:
                    vals = metric_grid[row][metric_name][line][repeat]

                    results.append(vals)

                results_grid[row][metric_name][line] = metric_fn(results, keys=repeats)

        if x_key is not None or x_vals is not None:
            for metric_name in metric_grid[row].keys():
                for line in metric_grid[row][metric_name].keys():
                    if x_key is not None:
                        results_grid[row][metric_name][line]["x"] = results_grid[row][
                            x_key
                        ][line]["center"]
                    elif x_vals is not None:
                        results_grid[row][metric_name][line]["x"] = x_vals

    return results_grid


def load_and_clean_experiments(
    exp_configs: List[Dict[str, Any]],
    results_dir: Union[str, List[str]],
    metrics: List[str],
    row_key: Union[Any, Iterator[Any]],
    line_key: Union[Any, Iterator[Any]],
    repeat_key: Union[Any, Iterator[Any]],
    metric_fn: Callable = utils.final_metrics,
    keep: List[Tuple[Any, Any]] = [],
    remove: List[Tuple[Any, Any]] = [],
    filter_fn: Optional[Callable] = None,
    transform_fn: Optional[Callable] = None,
    processing_fns: List[
        Callable[[Dict[str, np.ndarray], Tuple], Dict[str, np.ndarray]]
    ] = [],
    x_key: Optional[str] = None,
    x_vals: Optional[Union[List, np.ndarray]] = None,
):
    """Load and clean a grid of experiments according to the past parameters.
    This is a convenience function which composes other built-ins.
    :param exp_configs: list of experiment configuration objects. These will be expanded, filtered, and cleaned.
    :param results_dir: a base directory of list of base directories where the results are stored.
    :param metrics: list of strings identifying different metrics.
    :param row_key: key (or iterable of keys) for which distinct values in the experiment dictionaries are to be split into different rows.
    :param line_key: key (or iterable of keys) for which distinct values in the experiment dictionaries are to be split into different lines.
    :param line_key: key (or iterable of keys) for which distinct values in the experiment dictionaries are to be *averaged* over in the plot.
    :param metric_fn: (optional) function to process metrics after they are loaded.
    :param keep: A list of key-value pairs to retain with the form `[(key, values)]`. Each `key` is either a singleton
        key for the top-level dictionary or an iterable of keys indexing into nested dictionaries. `values` is either
        singleton or list of values.
    :param remove: A list of key-value pairs to filter with the form `[(key, values)]`. Arguments should taken the same
        form as `keep`.
    :param filter_fn: An additional filter to run on each dictionary.
    :param transform_fn: a transformation function to call on the complete metric grid.
    :param processing_fns: a list of functions to be called on the leafs of the loaded experiment grid.
        Order matters.
    :param x_key: a key which indexes into the values which should be used as the x-axis.
    :param x_vals: x-axis values. Cannot be supplied at the same time as 'x_key'.
    """

    exp_list = configs.expand_config_list(exp_configs)
    filtered_exp_list = configs.filter_dict_list(
        exp_list,
        keep=keep,
        remove=remove,
        filter_fn=filter_fn,
    )

    # make sure we load the metric associated with the x-axis.
    added_x = False

    if x_key is not None and x_key not in metrics:
        added_x = True
        metrics = metrics + [x_key]

    exp_grid = configs.make_metric_grid(
        filtered_exp_list, metrics, row_key, line_key, repeat_key
    )

    def call_on_fn(exp, key):
        for fn in processing_fns:
            exp = fn(exp, key)
        return exp

    metric_grid = load_metric_grid(
        exp_grid,
        results_dir,
        processing_fn=call_on_fn,
    )

    if transform_fn is not None:
        metric_grid = transform_fn(metric_grid)

    metric_grid = compute_metrics(
        metric_grid,
        metric_fn=metric_fn,
        x_key=x_key,
        x_vals=x_vals,
    )

    # drop x if necessary
    if added_x:
        for row, val in metric_grid.items():
            # remove x_key
            val.pop(x_key)

    return metric_grid
