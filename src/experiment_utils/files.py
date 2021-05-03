"""
Utilities for working with files.
"""

import os
import pickle as pkl
from copy import deepcopy
from typing import Dict, Any, Callable, Optional, cast

import torch

from experiment_utils.configs import hash_dict


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

    hash_id = hash_dict(exp_dict)
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
    exp_dict: dict,
    results_dir: str = "results",
    load_metrics: bool = False,
    load_model: bool = False,
) -> Dict[str, Any]:
    """Load results of the experiment corresponding to the given dictionary.
    :param exp_dict: experiment dictionary.
    :param results_dir: base directory for experimental results.
    :param load_metrics: whether or not to load metrics from the experiment.
    :param load_model: whether or not to load a model associated with the experiment.
    :returns: dict containing results. It is indexed by 'return_value' and (optionally) 'metrics', 'model'.
    """

    hash_id = hash_dict(exp_dict)
    path = os.path.join(results_dir, hash_id)
    results = {}

    if not os.path.exists(path):
        raise ValueError(f"Cannot find experiment in {results_dir}!")

    with open(os.path.join(path, "return_value.pkl"), "rb") as f:
        results["return_value"] = pkl.load(f)

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
    grid: dict, results_dir: str = "results", metric_fn: Callable = None
) -> dict:
    """Load metrics according to a supplied grid of experiment dictionaries.
    :param grid: a grid of experiment dictionaries. See 'configs.make_grid'.
    :param results_dir: base directory for experimental results.
    :param metric_fn: (optional) function to process metrics after they are loaded.
    :returns: nested dictionary with the same "shape" as 'grid' containing the loaded metrics.
    """

    results_grid = deepcopy(grid)
    if metric_fn is None:

        def metric_fn(x):
            return x

    for row in grid.keys():
        for col in grid[row].keys():
            for line in grid[row][col].keys():
                results = load_experiment(
                    grid[row][col][line],
                    results_dir,
                    load_metrics=True,
                    load_model=False,
                )
                results_grid[row][col][line] = metric_fn(results["metrics"])

    return results_grid
