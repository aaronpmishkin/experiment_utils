"""
Utilities for running experiments.
"""

import os
import pickle as pkl
from logging import Logger
from typing import Callable, Dict, Any, Tuple

from experiment_utils.configs import hash_dict
from experiment_utils.files import save_experiment


def run_or_load(
    logger: Logger,
    exp_dict: dict,
    run_fn: Callable[[Logger, Dict[str, Any], str], Tuple[Any, Any, Dict[str, Any]]],
    data_dir: str = "data",
    results_dir: str = "results",
    force_rerun: bool = False,
    save_return: bool = True,
):
    """Run and experiment or load the results if they already exist.
    :param logger: an instance of logging.Logger for use logging the experiment run.
    :param exp_dict: dictionary with experiment parameters.
    :param run_fn: callable that executes the desired experiment.
    :param data_dir: (optional) path to data files for experiment.
    :param results_dir: (optional) path to location where results are/will be saved.
    :param force_rerun: (optional) rerun experiment regardless of files on disk.
    :param save_return: (optional) save return value to `path`.
    :returns: the output of running the experiment, which is 'run_fn(logger, exp_dict)'.
    """

    path = os.path.join(results_dir, hash_dict(exp_dict), "metrics.pkl")

    if os.path.exists(path) and not force_rerun:
        logger.info("Loading results.")
        with open(path, "rb") as f:
            results = pkl.load(f)
    else:
        logger.info("Running.")
        results, model, metrics = run_fn(logger, exp_dict, data_dir)

        logger.info("Complete.")

        if save_return:
            logger.info("Saving results.")
            # save the experiment to disk.
            save_experiment(exp_dict, results_dir, results, metrics, model)

    return results
