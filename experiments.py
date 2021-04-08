"""
Utilities for running experiments.
"""

import os
import pickle as pkl

from configs import hash_dict


def run_or_load(logger, exp_dict, run_fn, extract_fn=None, dest_dir="results", force_rerun=False, save_return=True):
    """ Run and experiment or load the results if they already exist.
    :param logger: logging.Logger object for use logging the experiment run.
    :param exp_dict: dictionary with experiment parameters.
    :param run_fn: callable that executes the desired experiment.
    :param extract_fn: (optional) callable that processes the results of run_fn.
    :param path: (optional) path to location where results are/will be saved.
    :param force_rerun: (optional) rerun experiment regardless of files on disk.
    :param save_return: (optional) save return value to `path`.
    :returns: `extract_fn(run_fn())` or `run_fn()` if `extract_fn` is None.
    """

    path = os.path.join(hash_dict(dest_dir, exp_dict))

    if os.path.exists(path) and not force_rerun:
        logger.info("Loading results...")
        with open(path, "rb") as f:
            results = pkl.load(f)
    else:
        logger.info("Running...")
        results = run_fn(exp_dict)

        logger.info("Complete. Extracting results...")
        if extract_fn is not None:
            results = extract_fn(results)

        if save_return:
            logger.info("Saving results...")

            # make sure save path exists
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, "return_value.pkl"), "wb") as f:
                pkl.dump(results, f)

    return results
