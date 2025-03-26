"""
Utilities for working with experiment dictionaries.
"""

from __future__ import annotations
import os
import hashlib
from collections.abc import Callable, Iterator
from collections import defaultdict
from itertools import product
from functools import reduce
from copy import deepcopy
from typing import Any, cast

from .utils import as_list


def hash_dict(exp_dict: dict) -> str:
    """Hash an experiment dictionary into a unique id.

    The resulting id is usually used as a file-name for saving experiments.
    This strategy is adapted from
    `Haven AI <https://github.com/haven-ai/haven-ai>`_
    Params:
        exp_dict: An experiment dictionary.

    Returns:
        A unique id for the experiment.
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


def get_nested_value(exp_dict: dict, key: tuple | Any) -> Any:
    """Access a value in nested dictionary using a tuple of keys.

    Params:
        exp_dict: a (nested) dictionary.
        key: a singleton or iterable of keys for the nested dictionary.

    Returns:
        The value from the nested dictionary:
        `exp_dict[key[0]][key[1]...[key[-1]]`.
    """
    key = as_list(key)
    value = exp_dict
    for sub_key in key:
        value = value[sub_key]

    return value


def expand_config(config: dict | Any, recurse: bool = True) -> list[dict]:
    """Expand an configuration dictionary into a list of experiments.

    This function takes the Cartesian product of options from the config
    dictionary to produce a list of experiments.  Inspired by
    `Haven AI <https://github.com/haven-ai/haven-ai>`_

    Params:
        config: A (nested) dictionary with options for the experiments to run.
            Every list in the configuration is treated as a grid over which
            experiments are generated.  Use tuples (or some other iterable) to
            pass options which should not be expanded.
        recurse: whether or not to expand nested dictionaries to look for more
            configuration options.

    Returns:
        List of experiment dictionaries generated from `config`.
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


def expand_config_list(config_list: list[dict]) -> list[dict]:
    """Convenience function for expanding a list of experiment configurations.

    See `expand_config` for details on the expansion operation.

    Params:
        config_list: list of configuration objects to expand.

    Returns:
        The list of experiment dictionaries.
    """

    def expand_plus(acc, value):
        return acc + expand_config(value)

    return reduce(expand_plus, config_list, [])


def filter_dict_list(
    dict_list: list[dict],
    keep: list[tuple[Any, Any]] | None = None,
    remove: list[tuple[Any, Any]] | None = None,
    filter_fn: Callable | None = None,
) -> list[dict]:
    """Filter list of (nested) dictionaries based on key-value pairs.

    Params:
        dict_list: A list of dictionary objects. Each dictionary object may
            be composed of nested dictionaries.
        keep: A list of key-value pairs to retain with the form
            `[(key, values)]`. Each `key` is either a singleton key for the
            top-level dictionary or an iterable of keys indexing into nested
            dictionaries, while `values` is either singleton or list of values.
        remove: A list of key-value pairs to filter with the form
            `[(key, values)]`. Arguments should taken the same form as `keep`.
        filter_fn: (optional) An additional filter to run on each dictionary.

    Returns:
        Filtered list of dictionaries.

    """

    if keep is None:
        keep = []

    if remove is None:
        remove = []

    keys_to_check = list(zip(keep + remove, [True] * len(keep) + [False] * len(remove)))

    def key_filter(exp_dict):
        keep = True
        for entry, to_keep in keys_to_check:
            key, values_to_check = entry
            values_to_check = as_list(values_to_check)

            try:
                exp_value = get_nested_value(exp_dict, key)
            except KeyError:
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
    exp_list: list[dict],
    row_key: Any | Iterator[Any],
    col_key: Any | Iterator[Any],
    line_key: Any | Iterator[Any],
    repeat_key: Any | Iterator[Any],
    variation_key: Any | Iterator[Any],
) -> dict:
    """Convert an experiment list of into grid of experiment dictionaries.

    The grid of experiment dictionaries is organized in a hierarchy as
    rows -> columns -> lines -> repeats -> variations in preparation for plotting.
    Typically variations are optimizes over automatically.

    Params:
        exp_list: list of experiment dictionaries.
        row_key: key (or iterable of keys) for which distinct values in the
            experiment dictionaries are to be split into different rows.
        col_key: key (or iterable of keys) for which distinct values in the
            experiment dictionaries are to be split into different columns.
        line_key: key (or iterable of keys) for which distinct values in the
            experiment dictionaries are to be split into different lines.
        repeat_key: key (or iterable of keys) for which distinct values in the
            experiment dictionaries are to be *averaged* over in the plot.
        variation_key: key (or iterable of keys) for for which distinct values
            in the experiment dictionaries are to be optimized over.

    Returns:
        A nested dictionary whose first level is indexed by the unique values
        found at `row_key`, the second by the values at `col_key`, the
        third by values at `line_key`, the fourth by `repeat_key`, and the last
        by `variation_key`.

    """
    grid: dict = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    )

    for exp_dict in exp_list:
        row_val = (
            row_key(exp_dict)
            if callable(row_key)
            else get_nested_value(exp_dict, row_key)
        )
        col_val = (
            col_key(exp_dict)
            if callable(col_key)
            else get_nested_value(exp_dict, col_key)
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
        optimize_val = (
            variation_key(exp_dict)
            if callable(variation_key)
            else get_nested_value(exp_dict, variation_key)
        )
        # do *not* silently overwrite other experiments.
        if optimize_val in grid[row_val][col_val][line_val][repeat_val]:
            raise ValueError(
                (
                    f"Two experiment dicts match the same key-set: {row_val, col_val, line_val, repeat_val, optimize_val}\n"
                    f"{grid[row_val][col_val][line_val][repeat_val][optimize_val]}, \n\n"
                    f"{exp_dict}."
                )
            )

        grid[row_val][col_val][line_val][repeat_val][optimize_val] = exp_dict

    return grid


def make_metric_grid(
    exp_list: list[dict],
    metrics: list[str],
    row_key: Any | Iterator[Any],
    line_key: Any | Iterator[Any],
    repeat_key: Any | Iterator[Any],
    variation_key: Any | Iterator[Any],
) -> dict:
    """Convert an experiment list of into grid of experiment dictionaries.

    The difference between `make_metric_grid` and `make_grid` is where the
    values for columns are found. In this function, the unique columns come
    from the list of metrics while `make_grid` creates finds the columns from
    the unique values at `column_key` in the experiment dictionaries.

    Params:
        exp_list: list of experiment dictionaries.
        metrics: list of strings identifying different metrics.
        row_key: key (or iterable of keys) for which distinct values in the
            experiment dictionaries are to be split into different rows.
        line_key: key (or iterable of keys) for which distinct values in the
            experiment dictionaries are to be split into different lines.
        repeat_key: key (or iterable of keys) for which distinct values in the
        experiment dictionaries are to be *averaged* over in the plot.
        variation_key: key (or iterable of keys) for for which distinct values
            in the experiment dictionaries are to be optimized over.

    Returns:
        A nested dictionary whose first level is indexed by the unique values
        found at `row_key`, the second by the values from `metrics`, the
        third by values at `line_key`, the fourth by values at `repeat_key`,
        and the last by values at `variation_key`.

    """
    grid: dict = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    )

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
        optimize_val = (
            variation_key(exp_dict)
            if callable(variation_key)
            else get_nested_value(exp_dict, variation_key)
        )

        # do *not* silently overwrite other experiments.
        if optimize_val in grid[row_val][metric_name][line_val][repeat_val]:
            raise ValueError(
                (
                    f"Two experiment dicts match the same key-set: {row_val, metric_name, line_val, repeat_val, optimize_val}\n"
                    f"{grid[row_val][metric_name][line_val][repeat_val][optimize_val]}, \n\n"
                    f"{exp_dict}."
                )
            )

        grid[row_val][metric_name][line_val][repeat_val][optimize_val] = exp_dict

    return grid


def make_performance_profile_grid(
    exp_list: list[dict],
    row_key: Any | Iterator[Any],
    col_key: Any | Iterator[Any],
    problem_key: Any | Iterator[Any],
    line_key: Any | Iterator[Any],
) -> dict:
    """Convert an experiment list of into 3d grid of experiment dictionaries.

    The grid of experiment dictionaries is organized in a hierarchy as
    rows -> columns -> lines in preparation for plotting.

    Params:
        exp_list: list of experiment dictionaries.
        row_key: key (or iterable of keys) for which distinct values in the
            experiment dictionaries are to be split into different rows.
        col_key: key (or iterable of keys) for which distinct values in the
            experiment dictionaries are to be split into different columns.
        problem_key: key (or iterable of keys) for which distinct values in the
            experiment dictionaries are to be split into different lines.
        line_key: key (or iterable of keys) for which distinct values in the
            experiment dictionaries are to be split into different lines.

    Returns:
        A nested dictionary whose first level is indexed by the unique values
        found at `row_key`, the second by the values at `col_key`, the
        third by values at `problem_key`, and the fourth by the values at
        `line_key`.

    """
    grid: dict = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    )

    for exp_dict in exp_list:
        row_val = (
            row_key(exp_dict)
            if callable(row_key)
            else get_nested_value(exp_dict, row_key)
        )
        col_val = (
            col_key(exp_dict)
            if callable(col_key)
            else get_nested_value(exp_dict, col_key)
        )
        problem_val = (
            problem_key(exp_dict)
            if callable(problem_key)
            else get_nested_value(exp_dict, problem_key)
        )
        line_val = (
            line_key(exp_dict)
            if callable(line_key)
            else get_nested_value(exp_dict, line_key)
        )
        # do *not* silently overwrite other experiments.

        if line_val in grid[row_val][col_val][problem_val]:
            raise ValueError(
                (
                    f"Two experiment dicts match the same key-set: {row_val, col_val, problem_val, line_val} \n"
                    f"{grid[row_val][col_val][problem_val][line_val]}, \n\n"
                    f"{exp_dict}."
                )
            )

        grid[row_val][col_val][problem_val][line_val] = exp_dict

    return grid


def merge_grids(exp_grids: list[dict]) -> dict:
    """Merge a list of metric grids.

    Params:
        exp_grids: a list of experiment grids with layers of keys,
        (row, col, line, repeat).
    Returns:
        A merged experiment grid.
    """

    base_grid = deepcopy(exp_grids[0])
    for grid in exp_grids[1:]:
        for row in grid.keys():
            for col in grid[row].keys():
                for line in grid[row][col].keys():
                    if line in base_grid[row][col][line]:
                        raise ValueError(
                            (
                                f"Two experiment dicts match the same key-set: {row, col, line}"
                                f"\n {grid[row][col][line]}, \n \n"
                                f"{base_grid[row][col][line]}."
                            )
                        )
                    base_grid[row][col][line] = grid[row][col][line]
    return base_grid


def call_on_grid(exp_grid: dict, call_fn: Callable) -> dict:
    """Call 'call_fn' on the values stored in the leafs of an experiment grid.

    Params:
        exp_grid: a grid of experiments such as generated by 'make_grid'.
        call_fn: the function to call on each leaf value.
    :returns: a new experiment grid.
    """

    new_grid = deepcopy(exp_grid)
    for row in exp_grid.keys():
        for col in exp_grid[row].keys():
            for line in exp_grid[row][col].keys():
                for repeat in exp_grid[row][col][line].keys():
                    new_grid[row][col][line][repeat] = call_fn(
                        exp_grid[row][col][line][repeat],
                        (row, col, line, repeat),
                    )

    return new_grid


def leaves_to_roots(exp_grid: dict) -> dict:
    new_grid: dict = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    )

    for row in exp_grid.keys():
        for col in exp_grid[row].keys():
            for line in exp_grid[row][col].keys():
                for repeat in exp_grid[row][col][line].keys():
                    for variation in exp_grid[row][col][line][repeat].keys():
                        subdict = new_grid[variation][row][col][line]
                        val = exp_grid[row][col][line][repeat][variation]
                        subdict[repeat] = val

    return new_grid
