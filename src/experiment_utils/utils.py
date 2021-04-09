"""
Generic utilities.
"""
from functools import reduce, partial
from warnings import warn

import numpy as np


def quantile_metrics(metrics, quantiles=(0.25, 0.75)):
    """Compute quantiles and median of supplied run metrics. Statistics are computed *across* columns.
    :metrics: a list, np.ndarray, or dictionary containing run metrics.
    :quantiles: (optional) the quantiles to compute. Defaults to the first and third quartiles (0.25, 0.75).
    :returns: dictionary object with dict['center'] = mean, dict['upper'] = mean + std*k, dict['upper'] = mean - std*k.
    """
    metric_dict = {}
    assert quantiles[1] > quantiles[0]

    if type(metrics, dict):
        q_0_key = f"{quantiles[0]}_quantile"
        q_1_key = f"{quantiles[1]}_quantile"

        if q_0_key not in metrics or q_0_key not in metrics or "median" not in metrics:
            raise ValueError("If 'metrics' is a dictionary then it must contain pre-computed statistics, including median and desired quantiles.")
        q_0, q_1, median = metrics[q_0_key], metrics[q_1_key], metrics["median"]
    elif type(metrics, list):
        metrics_np = equalize_arrays(metrics)
        q_0, q_1 = np.quantile(metrics_np, quantiles[0], axis=0), np.quantile(metrics_np, quantiles[1], axis=0)
        median = np.median(metrics_np, axis=0)
    elif type(metrics, np.ndarray):
        q_0, q_1 = np.quantile(metrics_np, quantiles[0], axis=0), np.quantile(metrics_np, quantiles[1], axis=0)
        median = np.median(metrics_np, axis=0)
    else:
        raise ValueError(f"Cannot interpret metrics of type {type(metrics)}!")

    metric_dict['center'] = median
    metric_dict['upper'] = q_1
    metric_dict['lower'] = q_0

    return metric_dict


def std_dev_metrics(metrics, k=1):
    """Compute standard mean and deviation of supplied run metrics. Statistics are computed *across* columns.
    :metrics: a list, np.ndarray, or dictionary containing run metrics.
    :param k: number of standard deviations for computer 'upper' and 'lower' error bounds.
    :returns: dictionary object with dict['center'] = mean, dict['upper'] = mean + std*k, dict['upper'] = mean - std*k.
    """
    metric_dict = {}
    if type(metrics, dict):
        if "std" not in metrics or "mean" not in metrics:
            raise ValueError("If 'metrics' is a dictionary then it must contain pre-computed statistics, including mean and standard deviation.")
        std, mean = metrics["std"], metrics["mean"]
    elif type(metrics, list):
        metrics_np = equalize_arrays(metrics)
        std, mean = np.std(metrics_np, axis=0), np.mean(metrics_np, axis=0)
    elif type(metrics, np.ndarray):
        std, mean = np.std(metrics_np, axis=0), np.mean(metrics_np, axis=0)
    else:
        raise ValueError(f"Cannot interpret metrics of type {type(metrics)}!")

    metric_dict['center'] = mean
    metric_dict['upper'] = mean + std * k
    metric_dict['lower'] = mean - std * k

    if np.any(metrics['lower'] < 0):
        warn("Negative values encountered when computing lower error bounds. Consider using ")

    return metric_dict


def equalize_arrays(array_list):
    """Equalize the length of lists or np.ndarray objects inside a list by extending final values.
    :param array_list: list of arrays (or lists) to extend.
    :returns: np.ndarray object composed of extended lists.
    """
    max_length = reduce(lambda acc, x: max(acc, len(x)), array_list, 0)
    return np.array(list(map(partial(pad, length=max_length), array_list)))


def pad(array, length):
    """Pad 'array' with it's last value until it len(array) = length.
    :param array: 1-d np.ndarray or list.
    :param length: length to which the 'array' should be extended.
    """
    if length < len(array):
        raise ValueError("'length' must be at least the length of 'array'!")

    array_np = np.array(array)
    return np.concatenate([array_np, np.repeat([array_np[-1]], length - len(array_np))])
