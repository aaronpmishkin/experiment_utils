"""
Generic utilities.
"""
from __future__ import annotations
from copy import deepcopy
from typing import Any
from collections.abc import Callable
from functools import reduce
import logging

import numpy as np


def as_list(x: Any) -> list[Any]:
    """Wrap argument into a list if it is not iterable.

    Params:
         x: a (potential) singleton to wrap in a list.

    Returns:
        [x] if x is not iterable and x if it is.
    """
    # don't treat strings as iterables.
    if isinstance(x, str):
        return [x]

    try:
        _ = iter(x)
        return x
    except TypeError:
        return [x]


def quantile_metrics(
    metrics: list | dict | np.ndarray,
    quantiles: tuple[float, float] = (0.25, 0.75),
    metric_name: str | None = None,
) -> dict[str, np.ndarray]:
    """Compute quantiles and median of supplied run metrics.

    Statistics are computed for each column of the nested dictionary
    and are computed from the individual repeats.

    Params:
        metrics: a list or `np.ndarray` containing run metrics.
        quantiles: (optional) the quantiles to compute. Defaults to the first
            and third quartiles (0.25, 0.75).
        metric_name: the name of the metric.

    Returns:
        Dictionary object with `'upper'`, `'center'`, and `'lower'` keys for
        the upper, central, and lower and measures of tendency, respectively.
    """
    metric_dict = {}
    assert quantiles[1] > quantiles[0]

    if isinstance(metrics, list):
        value = None
        if metric_name == "time":
            value = 0.0
        metrics_np = equalize_arrays(metrics, value=value)

        if metric_name == "time":
            metrics_np = np.cumsum(metrics_np, axis=-1)

    elif isinstance(metrics, np.ndarray):
        metrics_np = metrics
    else:
        raise ValueError(f"Cannot interpret metrics of type {type(metrics)}!")

    q_0, q_1 = np.quantile(metrics_np, quantiles[0], axis=0), np.quantile(
        metrics_np, quantiles[1], axis=0
    )
    median = np.median(metrics_np, axis=0)

    metric_dict["center"] = median
    metric_dict["upper"] = q_1
    metric_dict["lower"] = q_0

    return metric_dict


def std_dev_metrics(
    metrics: list | dict | np.ndarray,
    k: int = 1,
    metric_name: str | None = None,
) -> dict[str, np.ndarray]:
    """Compute standard mean and deviation of supplied run metrics.

    Statistics are computed for each column of the nested dictionary
    and are computed from the individual repeats.

    Params:
        metrics: a list or `np.ndarray` containing run metrics.
        k: number of standard deviations for computer 'upper' and 'lower'
            error bounds.
        metric_name: the name of the metric.

    Returns:
        Dictionary object where `dict['center']` is the mean, `dict['upper']`
        is `mean + std*k` and `dict['lower']` is `mean - std*k`.
    """
    metric_dict = {}
    if isinstance(metrics, list):
        value = None
        if metric_name == "time":
            value = 0.0
        metrics_np = equalize_arrays(metrics, value=value)

        if metric_name == "time":
            metrics_np = np.cumsum(metrics_np, axis=-1)

        metrics_np = equalize_arrays(metrics)
    elif isinstance(metrics, np.ndarray):
        metrics_np = metrics
    else:
        raise ValueError(f"Cannot interpret metrics of type {type(metrics)}!")

    std, mean = np.std(metrics_np, axis=0), np.mean(metrics_np, axis=0)

    metric_dict["center"] = mean
    metric_dict["upper"] = mean + std * k
    metric_dict["lower"] = mean - std * k

    return metric_dict


def final_metrics(
    metrics: list | dict | np.ndarray,
    window: int = 1,
    metric_name: str | None = None,
) -> dict[str, np.ndarray]:
    """Extract final metrics from each run.

    Statistics are computed for each column of the nested dictionary
    and are computed from the individual repeats.

    Params:
        metrics: a list, np.ndarray, or dictionary containing run metrics.
        window: length of window over which to take the max, min, and median.
        metric_name (optional):

    Returns:
        Dictionary object where the `upper`, `center` and `lower` keys index
        the maximum, median, and minimum metric values over the window.
    """
    metric_dict = {}
    if isinstance(metrics, (list, np.ndarray)):
        upper, lower, center = [], [], []
        for run in metrics:
            run_np = np.array(run)
            upper.append(np.max(run_np[-window:]))
            lower.append(np.min(run_np[-window:]))
            center.append(np.median(run_np[-window:]))
    else:
        raise ValueError(f"Cannot interpret metrics of type {type(metrics)}!")

    metric_dict["center"] = np.array(center)
    metric_dict["upper"] = np.array(upper)
    metric_dict["lower"] = np.array(lower)

    return metric_dict


def equalize_arrays(
    array_list: list[list | np.ndarray],
    value: float | None = None,
) -> np.ndarray:
    """Equalize the length of lists or `np.ndarray` objects inside a list.

    Equalization is achieved by padding each list/array with a single value,
    which defaults to the last value in each list/array.


    Params:
        array_list: list of arrays (or lists) to extend.
        value: (optional) a specific value to use when padding each list.

    Returns:
        An `np.ndarray` matrix obtained by stacking the equalized lists/arrays.
    """
    max_length = reduce(lambda acc, x: max(acc, len(x)), array_list, 0)
    return np.array([pad(v, length=max_length, value=value) for v in array_list])


def pad(array: list | np.ndarray, length: int, value=None) -> np.ndarray:
    """Pad an array until `len(array) = length`.

    Params:
        array: 1-d `np.ndarray` or list.
        length: length to which the 'array' should be extended.

    Returns:
        Padded list/array in `np.ndarray` format.
    """

    if length == 0:
        return np.array([1.0])

    if length < len(array):
        raise ValueError("'length' must be at least the length of 'array'!")

    if isinstance(array, list) and isinstance(array[0], list):
        # handle recursive case
        array = equalize_arrays(array)

    array_np = np.array(array)

    if value is None:
        value = array_np[-1]

    return np.concatenate([array_np, np.repeat([value], length - len(array_np))])


def normalize(filter_func: Callable):
    def closure(vals: list | np.ndarray, key: tuple[Any]) -> list | np.ndarray:
        if filter_func(key):
            vals = vals - vals[0]

        return vals

    return closure


def drop_start(start: int, filter_func: Callable):
    def closure(vals: list | np.ndarray, key: tuple[Any]) -> list | np.ndarray:
        if filter_func(key) and len(vals) > start:
            vals = vals[start:]

        return vals

    return closure


def extend(length: int, filter_func: Callable):
    def closure(vals: list | np.ndarray, key: tuple[Any]) -> list | np.ndarray:
        if filter_func(key) and len(vals) < length:
            vals = pad(vals, length)

        return vals

    return closure


def cum_sum(filter_func: Callable):
    def closure(vals: list | np.ndarray, key: tuple[Any]) -> list | np.ndarray:
        if filter_func(key):
            vals = np.cumsum(vals)

        return vals

    return closure


def total_f_evals(filter_func: Callable):
    def closure(vals: list | np.ndarray, key: tuple[Any]) -> list | np.ndarray:
        if filter_func(key):
            vals = np.cumsum(vals + 1.0)

        return vals

    return closure


# data transformation functions


def replace_x_axis(metric_grid):
    results_grid = deepcopy(metric_grid)

    for row in metric_grid.keys():
        for metric_name in metric_grid[row].keys():
            for line in metric_grid[row][metric_name].keys():
                results = {}
                repeats = metric_grid[row][metric_name][line].keys()

                for x_key, repeat_key in repeats:
                    vals = results_grid[row][metric_name][line][(x_key, repeat_key)]

                    results[repeat_key] = results.get(repeat_key, []) + [
                        (x_key, vals[-1])
                    ]

                # make sure the order is correct.
                for key, val in results.items():
                    results[key] = np.array(
                        [final for lam, final in sorted(val, key=lambda x: x[0])]
                    )

                results_grid[row][metric_name][line] = results

    return results_grid


# Logging #


def get_logger(
    name: str,
    verbose: bool = False,
    debug: bool = False,
    log_file: str | None = None,
) -> logging.Logger:
    """Construct a logging.Logger instance with an appropriate configuration.

    Params:
        name: name for the Logger instance.
        verbose: (optional) whether or not the logger should print verbosely
            (ie. at the `INFO` level).
        debug: (optional) whether or not the logger should print in debug mode
            (ie. at the `DEBUG` level).
        log_file: (optional) path to a file where the log should be stored. The
            log is printed to `stdout` when `None`.

     Returns:
        Instance of logging.Logger.
    """

    level = logging.WARNING
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO

    logging.basicConfig(level=level, filename=log_file)
    logger = logging.getLogger(name)
    logging.root.setLevel(level)
    logger.setLevel(level)
    return logger
