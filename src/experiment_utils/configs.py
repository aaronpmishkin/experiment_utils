"""
Utilities for working with experiment dictionaries.
"""
import os
import hashlib
from collections import defaultdict
from itertools import product
from functools import reduce
from copy import deepcopy
from typing import Any, List, Tuple, Union, Optional, Callable, Iterator, cast, Dict

from experiment_utils.utils import as_list


def hash_dict(exp_dict: Dict) -> str:
    """Hash an experiment dictionary into a unique id.
    Can be used as a file-name. Adapted from Haven AI (https://github.com/haven-ai/haven-ai).
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


def get_nested_value(exp_dict: Dict, key: Union[Tuple, Any]) -> Any:
    """Access value in nested dictionary.
    :param exp_dict: a (nested) dictionary.
    :param key: a singleton or iterable of keys for the nested dictionary.
    :returns: the value from the nested dictionary: `exp_dict[key[0]][key[1]...[key[-1]]`.
    """
    key = as_list(key)
    value = exp_dict
    for sub_key in key:
        value = value[sub_key]

    return value


def expand_config(config: Union[Dict, Any], recurse: bool = True) -> List[Dict]:
    """Expand an experiment configuration into a list of experiment dictionaries.
    Inspired by Haven AI [https://github.com/haven-ai/haven-ai/]
    :param config: A (nested) dictionary with options for the experiments to run.
        Every list in the configuration is treated as a grid over which experiments are generated.
        Use tuples (or some other iterable) to pass options which should not be expanded.
    :param recurse: whether or not to expand nested dictionaries.
    :returns: list of experiment dictionaries generated from `config`.
    """
    if not isinstance(config, dict):
        return [config]

    config = cast(dict, config)

    # deep copy to ensure that configuration objects are all different.
    exp_config_copy = deepcopy(config)

    # sort to ensure consistent order.
    for key in sorted(exp_config_copy.keys()):
        if isinstance(exp_config_copy[key], dict) and recurse:
            exp_config_copy[key] = expand_config(exp_config_copy[key])
        elif isinstance(exp_config_copy[key], list) and recurse:
            exp_config_copy[key] = reduce(
                lambda acc, v: acc + expand_config(v), exp_config_copy[key], []
            )
        elif not isinstance(exp_config_copy[key], list):
            exp_config_copy[key] = [exp_config_copy[key]]

    # Create the Cartesian product
    exp_list = [
        dict(zip(exp_config_copy.keys(), values))
        for values in product(*exp_config_copy.values())
    ]

    return exp_list


def expand_config_list(config_list: List[Dict]) -> List[Dict]:
    """Convenience function for expanding a list of experiment configurations.
    See `expand_config` for details on the expansion operation.
    :param config_list: list of configuration objects to expand.
    :returns: the list of experiment dictionaries.
    """

    def expand_plus(acc, value):
        return acc + expand_config(value)

    return reduce(expand_plus, config_list, [])


def filter_dict_list(
    dict_list: List[Dict],
    keep: List[Tuple[Any, Any]] = [],
    remove: List[Tuple[Any, Any]] = [],
    filter_fn: Optional[Callable] = None,
) -> List[Dict]:
    """Filter list of (nested) dictionaries based on key-value pairs.
    :param dict_list: A list of dictionary objects. Each dictionary object may be composed of nested dictionaries.
    :param keep: A list of key-value pairs to retain with the form `[(key, values)]`. Each `key` is either a singleton
        key for the top-level dictionary or an iterable of keys indexing into nested dictionaries. `values` is either
        singleton or list of values.
    :param remove: A list of key-value pairs to filter with the form `[(key, values)]`. Arguments should taken the same
        form as `keep`.
    :param filter_fn: (optional) An additional filter to run on each dictionary.
    :returns: Filtered list of dictionaries.

    """

    keys_to_check = list(zip(keep + remove, [True] * len(keep) + [False] * len(remove)))

    def key_filter(exp_dict):
        keep = True
        for entry, to_keep in keys_to_check:
            key, values_to_check = entry
            values_to_check = as_list(values_to_check)

            try:
                exp_value = get_nested_value(exp_dict, key)
            except:
                continue

            # keep or filter the experiment
            if to_keep:
                keep = keep and exp_value in values_to_check
            else:
                keep = keep and exp_value not in values_to_check

        return keep

    if filter_fn is not None:

        def final_filter(exp_dict):
            return key_filter(exp_dict) and filter_fn(exp_dict)

    else:
        final_filter = key_filter

    return list(filter(final_filter, dict_list))


def make_grid(
    exp_list: List[Dict],
    row_key: Union[Any, Iterator[Any]],
    col_key: Union[Any, Iterator[Any]],
    line_key: Union[Any, Iterator[Any]],
    repeat_key: Union[Any, Iterator[Any]],
) -> Dict:
    """Convert an experiment list of into grid of experiment dictionaries in preparation for plotting.
    :param exp_list: list of experiment dictionaries. These should be expanded and filtered.
    :param row_key: key (or iterable of keys) for which distinct values in the experiment dictionaries are to be split into different rows.
    :param col_key: key (or iterable of keys) for which distinct values in the experiment dictionaries are to be split into different columns.
    :param line_key: key (or iterable of keys) for which distinct values in the experiment dictionaries are to be split into different lines.
    :param line_key: key (or iterable of keys) for which distinct values in the experiment dictionaries are to be *averaged* over in the plot.
    :returns: a nested dictionary whose first level is indexed by the unique values found at 'row_key', the second by the values at 'col_key', and the third by values at 'line_key'.

    """
    grid: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for exp_dict in exp_list:

        row_val = (
            row_key(exp_dict)
            if callable(row_key)
            else get_nested_value(exp_dict, row_key)
        )
        line_val = (
            line_key(exp_dict)
            if callable(line_key)
            else get_nested_value(exp_dict, line_key)
        )
        col_val = (
            col_key(exp_dict)
            if callable(col_key)
            else get_nested_value(exp_dict, col_key)
        )
        repeat_val = (
            repeat_key(exp_dict)
            if callable(repeat_key)
            else get_nested_value(exp_dict, repeat_key)
        )

        # do *not* silently overwrite other experiments.
        if repeat_val in grid[row_val][col_key][line_val]:
            raise ValueError(
                f"Two experiment dicts match the same key-set: \n {grid[row_val][col_key][line_val][repeat_val]}, \n \n {exp_dict}."
            )

        grid[row_val][col_val][line_val][repeat_val] = exp_dict

    return grid


def make_metric_grid(
    exp_list: List[Dict],
    metrics: List[str],
    row_key: Union[Any, Iterator[Any]],
    line_key: Union[Any, Iterator[Any]],
    repeat_key: Union[Any, Iterator[Any]],
) -> Dict:
    """Convert an experiment list of into grid of experiment dictionaries in preparation for plotting.
    :param exp_list: list of experiment dictionaries. These should be expanded and filtered.
    :param metrics: list of strings identifying different metrics.
    :param row_key: key (or iterable of keys) for which distinct values in the experiment dictionaries are to be split into different rows.
    :param line_key: key (or iterable of keys) for which distinct values in the experiment dictionaries are to be split into different lines.
    :param line_key: key (or iterable of keys) for which distinct values in the experiment dictionaries are to be *averaged* over in the plot.
    :returns: a nested dictionary whose first level is indexed by the unique values found at 'row_key', the second by the values at 'col_key', and the third by values at 'line_key'.

    """
    grid: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for exp_dict, metric_name in product(exp_list, metrics):

        row_val = (
            row_key(exp_dict)
            if callable(row_key)
            else get_nested_value(exp_dict, row_key)
        )
        line_val = (
            line_key(exp_dict)
            if callable(line_key)
            else get_nested_value(exp_dict, line_key)
        )
        repeat_val = (
            repeat_key(exp_dict)
            if callable(repeat_key)
            else get_nested_value(exp_dict, repeat_key)
        )

        # do *not* silently overwrite other experiments.
        if repeat_val in grid[row_val][metric_name][line_val]:
            raise ValueError(
                f"Two experiment dicts match the same key-set: \n {grid[row_val][metric_name][line_val][repeat_val]}, \n \n {exp_dict}."
            )

        grid[row_val][metric_name][line_val][repeat_val] = exp_dict

    return grid


def merge_grids(exp_grids: List[Dict]) -> Dict:
    """Merge a list of metric grids.
    :param exp_grids: a list of experiment grids with layers of keys,
        (row, col, line).
    :returns: a merged experiment grid.
    """

    base_grid = deepcopy(exp_grids[0])
    for grid in exp_grids[1:]:
        for row in grid.keys():
            for col in grid[row].keys():
                for line in grid[row][col].keys():
                    if line in base_grid[row][col][line]:
                        raise ValueError(
                            f"Two experiment dicts match the same key-set: \n {grid[row][col][line]}, \n \n {base_grid[row][col][line]}."
                        )
                    else:
                        base_grid[row][col][line] = grid[row][col][line]

    return base_grid


def call_on_grid(exp_grid: Dict, call_fn: Callable) -> Dict:
    """Call 'call_fn' on the values stored in the leafs of an experiment grid.
    :param exp_grid: a grid of experiments such as generated by 'make_grid'.
    :param call_fn: the function to call on each leaf value.
    :returns: a new experiment grid.
    """

    new_grid = deepcopy(exp_grid)
    for row in exp_grid.keys():
        for col in exp_grid[row].keys():
            for line in exp_grid[row][col].keys():
                for repeat in exp_grid[row][col][line].keys():
                    new_grid[row][col][line][repeat] = call_fn(
                        exp_grid[row][col][line][repeat], (row, col, line, repeat)
                    )

    return new_grid
