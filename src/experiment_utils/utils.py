"""
Generic utilities.
"""
from functools import reduce, partial
from warnings import warn
import logging
from typing import List, Any, Tuple, Union, Dict

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
) -> Dict[str, np.ndarray]:
    """Compute quantiles and median of supplied run metrics. Statistics are computed *across* columns.
    :metrics: a list, np.ndarray, or dictionary containing run metrics.
    :quantiles: (optional) the quantiles to compute. Defaults to the first and third quartiles (0.25, 0.75).
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
    metrics: Union[List, dict, np.ndarray], k: int = 1
) -> Dict[str, np.ndarray]:
    """Compute standard mean and deviation of supplied run metrics. Statistics are computed *across* columns.
    :metrics: a list, np.ndarray, or dictionary containing run metrics.
    :param k: number of standard deviations for computer 'upper' and 'lower' error bounds.
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
    if length < len(array):
        raise ValueError("'length' must be at least the length of 'array'!")

    array_np = np.array(array)
    return np.concatenate([array_np, np.repeat([array_np[-1]], length - len(array_np))])


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
    return logging.getLogger(name)
