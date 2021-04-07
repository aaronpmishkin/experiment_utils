"""
Utilities for working with files.
"""

import os
import pickle as pkl

import torch

from configs import hash_dict


def load_experiment(exp_dict, base_dir="results", load_metrics=True, load_model=False):
    """Load results of the experiment corresponding to the given dictionary.
    :param exp_dict: An experiment dictionary.
    :param base_dir: Base directory for experimental results.
    :param load_metrics: whether or not to load metrics from the experiment.
    :param load_model: whether or not to load a model associated with the experiment.
    :returns: A unique id for the experiment
    """

    if not load_metrics and not load_model:
        raise ValueError(
            "At least one of 'load_metrics' and 'load_model' should be 'True'"
        )

    hash_id = hash_dict(exp_dict)
    path = os.path.joint(base_dir, hash_id)
    results = {}

    if not os.path.exists(path):
        raise ValueError(f"Cannot find experiment in {base_dir}!")

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


# TODO: finish this
def load_grid(arg1):
    """TODO: Docstring for load_grid.

    :arg1: TODO
    :returns: TODO

    """
    pass
