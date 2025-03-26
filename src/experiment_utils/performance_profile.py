"""
Utilities for generating performance profile plots.
"""

import os
from typing import Callable, Dict, Any, List
from operator import itemgetter
from collections import defaultdict
from copy import deepcopy
import math

import numpy as np

from experiment_utils import files


def compute_success_ratios(
    exp_grid: dict,
    compute_xy_values: Callable,
    best_target: dict[Any, list],
    tol: float,
    maximize_target: bool = False,
) -> dict[str, Any]:
    results: Dict[str, Any] = defaultdict(lambda: defaultdict(list))
    best_results = defaultdict(list)

    for problem in exp_grid.keys():
        for line in exp_grid[problem].keys():
            x, target = exp_grid[problem][line]

            x_value, success = compute_xy_values(
                target,
                x,
                best_target[problem],
                tol,
                maximize_target,
            )

            results[line][problem].append([x_value, success])

    for line in results.keys():
        for problem in results[line].keys():
            # find the best run by minimizing over x-axis
            successes = list(filter(itemgetter(1), results[line][problem]))
            if len(successes) > 0:
                best_results[line].append(min(successes, key=itemgetter(0)))
            else:
                best_results[line].append(
                    max(results[line][problem], key=itemgetter(0))
                )

    n_problems = max([len(best_results[key]) for key in best_results.keys()])

    final_results = {}
    for line in best_results.keys():
        ordered = list(sorted(best_results[line], key=itemgetter(0)))
        values = np.array(ordered).T
        cumulative_successes = np.cumsum(values[1]) / n_problems
        final_results[line] = {"center": cumulative_successes, "x": values[0]}

    return best_results


def compute_xy_values_tol(target, x, best_target, tol, maximize_target=False):
    start_target = target[0]

    if maximize_target:
        min_target = np.min(target)
        if np.isnan(min_target):
            max_target = -np.inf
    else:
        max_target = np.max(target)
        if np.isnan(max_target):
            max_target = np.inf

    rel_diff = np.abs((target - best_target) / (start_target - best_target))

    thresholded_tols = rel_diff <= tol
    success = np.any(thresholded_tols)

    best_ind = np.argmax(thresholded_tols)

    x_coord = x[-1]
    if success:
        x_coord = x[best_ind]

    return x_coord, success
