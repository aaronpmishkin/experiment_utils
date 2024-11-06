"""
Utilities for running experiments.
"""

from __future__ import annotations
from typing import Any
import os
import pickle as pkl
from logging import Logger
from collections.abc import Callable

from .configs import hash_dict
from .files import save_experiment


def run_or_load(
    logger: Logger,
    exp_dict: dict,
    run_fn: Callable[
        [Logger, dict[str, Any], str, str], tuple[Any, Any, dict[str, Any]]
    ],
    data_dir: str = "data",
    results_dir: str = "results",
    force_rerun: bool = False,
    save_return: bool = True,
):
    """Run and experiment or load the results if they already exist.

    Params:
        logger: an instance of logging.Logger for use logging the experiment
        run.
        exp_dict: dictionary with experiment parameters.
        run_fn: callable that executes the desired experiment.
        data_dir: (optional) path to data files for experiment.
        results_dir: (optional) path to location where results are/will be
        saved.
        force_rerun: (optional) rerun experiment regardless of files on disk.
        save_return: (optional) save return value to `path`.

    Returns:
        the output of running the experiment.
    """

    path = os.path.join(results_dir, hash_dict(exp_dict), "metrics.pkl")

    if os.path.exists(path) and not force_rerun:
        logger.info("Loading results.")
        with open(path, "rb") as f:
            results = pkl.load(f)
    else:
        logger.info("Running.")

        results, model, metrics = run_fn(
            logger,
            exp_dict,
            data_dir,
            results_dir,
        )

        logger.info("Complete.")

        if save_return:
            logger.info("Saving results.")
            # save the experiment to disk.
            save_experiment(exp_dict, results_dir, results, metrics, model)

    return results


def filter_experiment_list(experiment_list, results_dir, force_rerun=False):
    filtered_list = []

    for exp_dict in experiment_list:
        path = os.path.join(results_dir, hash_dict(exp_dict), "metrics.pkl")

        if not os.path.exists(path) or force_rerun:
            filtered_list.append(exp_dict)

    return filtered_list
