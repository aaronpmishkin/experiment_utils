"""
Generic utilities.
"""
from copy import deepcopy
from typing import List, Any, Tuple, Union, Dict, Callable, Optional
from functools import reduce, partial
from warnings import warn
import logging

import numpy as np


def as_list(x: Any) -> List[Any]:
    """Wrap argument into a list if it is not iterable.
    :param x: a (potential) singleton to wrap in a list.
    :returns: [x] if x is not iterable and x if it is.
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
    metrics: Union[List, dict, np.ndarray],
    quantiles: Tuple[float, float] = (0.25, 0.75),
    keys: Optional[List[Any]] = None,
) -> Dict[str, np.ndarray]:
    """Compute quantiles and median of supplied run metrics. Statistics are computed *across* columns.
    :param metrics: a list, np.ndarray, or dictionary containing run metrics.
    :param quantiles: (optional) the quantiles to compute. Defaults to the first and third quartiles (0.25, 0.75).
    :param keys: (optional) the keys associated with the each metric.
    :returns: dictionary object with dict['center'] = mean, dict['upper'] = mean + std*k, dict['upper'] = mean - std*k.
    """
    metric_dict = {}
    assert quantiles[1] > quantiles[0]

    if isinstance(metrics, dict):
        q_0_key = f"{quantiles[0]}_quantile"
        q_1_key = f"{quantiles[1]}_quantile"

        if q_0_key not in metrics or q_0_key not in metrics or "median" not in metrics:
            raise ValueError(
                "If 'metrics' is a dictionary then it must contain pre-computed statistics, including median and desired quantiles."
            )
        q_0, q_1, median = metrics[q_0_key], metrics[q_1_key], metrics["median"]
    elif isinstance(metrics, list):
        metrics_np = equalize_arrays(metrics)
        q_0, q_1 = np.quantile(metrics_np, quantiles[0], axis=0), np.quantile(
            metrics_np, quantiles[1], axis=0
        )
        median = np.median(metrics_np, axis=0)
    elif isinstance(metrics, np.ndarray):
        q_0, q_1 = np.quantile(metrics_np, quantiles[0], axis=0), np.quantile(
            metrics_np, quantiles[1], axis=0
        )
        median = np.median(metrics_np, axis=0)
    else:
        raise ValueError(f"Cannot interpret metrics of type {type(metrics)}!")

    metric_dict["center"] = median
    metric_dict["upper"] = q_1
    metric_dict["lower"] = q_0

    return metric_dict


def std_dev_metrics(
    metrics: Union[List, dict, np.ndarray],
    k: int = 1,
    keys: Optional[List[Any]] = None,
) -> Dict[str, np.ndarray]:
    """Compute standard mean and deviation of supplied run metrics. Statistics are computed *across* columns.
    :metrics: a list, np.ndarray, or dictionary containing run metrics.
    :param k: number of standard deviations for computer 'upper' and 'lower' error bounds.
    :param keys: (optional) the keys associated with the each metric.
    :returns: dictionary object with dict['center'] = mean, dict['upper'] = mean + std*k, dict['upper'] = mean - std*k.
    """
    metric_dict = {}
    if isinstance(metrics, dict):
        if "std" not in metrics or "mean" not in metrics:
            raise ValueError(
                "If 'metrics' is a dictionary then it must contain pre-computed statistics, including mean and standard deviation."
            )
        std, mean = metrics["std"], metrics["mean"]
    elif isinstance(metrics, list):
        metrics_np = equalize_arrays(metrics)
        std, mean = np.std(metrics_np, axis=0), np.mean(metrics_np, axis=0)
    elif isinstance(metrics, np.ndarray):
        std, mean = np.std(metrics_np, axis=0), np.mean(metrics_np, axis=0)
    else:
        raise ValueError(f"Cannot interpret metrics of type {type(metrics)}!")

    metric_dict["center"] = mean
    metric_dict["upper"] = mean + std * k
    metric_dict["lower"] = mean - std * k

    if np.any(metric_dict["lower"] < 0):
        warn(
            "Negative values encountered when computing lower error bounds. Consider using "
        )

    return metric_dict


def final_metrics(
    metrics: Union[List, dict, np.ndarray],
    window: int = 1,
    keys: Optional[List[Any]] = None,
) -> Dict[str, np.ndarray]:
    """Extract final run metrics. Statistics are computed *across* columns.
    :metrics: a list, np.ndarray, or dictionary containing run metrics.
    :param window: length of window over which to take the max, min, and median.
    :param keys: (optional) the keys associated with the each metric.
    :returns: dictionary object with dict['center'] = mean, dict['upper'] = mean + std*k, dict['upper'] = mean - std*k.
    """
    metric_dict = {}
    if isinstance(metrics, list) or isinstance(metrics, np.ndarray):
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

    if keys is not None:
        metric_dict["x"] = np.array(list(keys))

    return metric_dict


def equalize_arrays(array_list: List[Union[List, np.ndarray]]) -> np.ndarray:
    """Equalize the length of lists or np.ndarray objects inside a list by extending final values.
    :param array_list: list of arrays (or lists) to extend.
    :returns: np.ndarray object composed of extended lists.
    """
    max_length = reduce(lambda acc, x: max(acc, len(x)), array_list, 0)
    return np.array(list(map(partial(pad, length=max_length), array_list)))


def pad(array: Union[List, np.ndarray], length: int) -> np.ndarray:
    """Pad 'array' with it's last value until it len(array) = length.
    :param array: 1-d np.ndarray or list.
    :param length: length to which the 'array' should be extended.
    """

    if length == 0:
        return np.array([1.0])

    elif length < len(array):
        raise ValueError("'length' must be at least the length of 'array'!")

    array_np = np.array(array)
    return np.concatenate([array_np, np.repeat([array_np[-1]], length - len(array_np))])


def normalize(filter_func: Callable):
    def closure(
        vals: Union[list, np.ndarray], key: Tuple[Any]
    ) -> Union[list, np.ndarray]:
        if filter_func(key):
            vals = vals - vals[0]

        return vals

    return closure


def drop_start(start: int, filter_func: Callable):
    def closure(
        vals: Union[list, np.ndarray], key: Tuple[Any]
    ) -> Union[list, np.ndarray]:
        if filter_func(key) and len(vals) > start:
            vals = vals[start:]

        return vals

    return closure


def cum_sum(filter_func: Callable):
    def closure(
        vals: Union[list, np.ndarray], key: Tuple[Any]
    ) -> Union[list, np.ndarray]:
        if filter_func(key):
            vals = np.cumsum(vals)

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

                for (x_key, repeat_key) in repeats:
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
    name: str, verbose: bool = False, debug: bool = False, log_file: str = None
) -> logging.Logger:
    """Construct a logging.Logger instance with an appropriate configuration.
    :param name: name for the Logger instance.
    :param verbose: (optional) whether or not the logger should print verbosely (ie. at the INFO level).
        Defaults to False.
    :param debug: (optional) whether or not the logger should print in debug mode (ie. at the DEBUG level).
        Defaults to False.
    :param log_file: (optional) path to a file where the log should be stored. The log is printed to stdout when 'None'.
    :returns: instance of logging.Logger.
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
