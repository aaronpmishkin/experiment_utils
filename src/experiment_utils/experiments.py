"""
Utilities for running experiments.
"""

import os
import pickle as pkl

from experiment_utils.configs import hash_dict


def run_or_load(logger, exp_dict, run_fn, data_dir="data", results_dir="results", force_rerun=False, save_return=True):
    """ Run and experiment or load the results if they already exist.
    :param logger: an instance of logging.Logger for use logging the experiment run.
    :param exp_dict: dictionary with experiment parameters.
    :param run_fn: callable that executes the desired experiment.
    :param data_dir: (optional) path to data files for experiment.
    :param results_dir: (optional) path to location where results are/will be saved.
    :param force_rerun: (optional) rerun experiment regardless of files on disk.
    :param save_return: (optional) save return value to `path`.
    :returns: the output of running the experiment, which is 'run_fn(logger, exp_dict)'.
    """

    path = os.path.join(hash_dict(results_dir, exp_dict))

    if os.path.exists(path) and not force_rerun:
        logger.info("Loading results.")
        with open(path, "rb") as f:
            results = pkl.load(f)
    else:
        logger.info("Running...")
        results = run_fn(logger, exp_dict, data_dir)

        logger.info("Complete.")

        if save_return:
            logger.info("Saving results.")

            # make sure save path exists
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, "return_value.pkl"), "wb") as f:
                pkl.dump(results, f)

    return results
