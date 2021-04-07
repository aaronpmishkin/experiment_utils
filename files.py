"""
    Utilities for working with files.
"""

import os
import hashlib
import pickle as pkl

import torch


def hash_dict(exp_dict):
    """Hash an experiment dictionary into a unique id.
    Can be used as a file-name. Adapted from Haven (https://github.com/haven-ai/haven-ai).
    :param exp_dict: An experiment dictionary.
    :returns: A unique id for the experiment
    """
    dict2hash = ""
    if not isinstance(exp_dict, dict):
        raise ValueError("exp_dict is not a dict")

    for k in sorted(exp_dict.keys()):
        if isinstance(exp_dict[k], dict):
            v = hash_dict(exp_dict[k])
        else:
            v = exp_dict[k]

        dict2hash += os.path.join(str(k), str(v))

    hash_id = hashlib.md5(dict2hash.encode()).hexdigest()

    return hash_id


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
