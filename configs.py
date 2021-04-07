"""
General utilities
"""

import os
import hashlib
from itertools import product
from functools import reduce


def as_list(x):
    """Wrap argument into a list if it is not iterable.
    :param x: a (potential) singleton to wrap in a list.
    :returns: [x] if x is not iterable and x if it is.
    """
    try:
        _ = iter(x)
        return x
    except TypeError:
        return list(x)


def expand_config(config, recurse=True):
    """ Expand an experiment configuration into a list of experiment dictionaries.
    Adapted from Haven AI [https://github.com/haven-ai/haven-ai/]
    :param config: A (nested) dictionary with options for the experiments to run.
        Every list in the configuration is treated as a grid over which experiments are generated.
        Use tuples (or some other iterable) to pass options which should not be expanded.
    :param recurse: whether or not to expand nested dictionaries.
    :returns: list of experiment dictionaries generated from `config`.

    """
    exp_config_copy = config.copy()

    for key, value in exp_config_copy.items():
        if type(value, dict) and recurse:
            exp_config_copy[key] = expand_config(value)
        if not type(exp_config_copy[key], list):
            exp_config_copy[key] = [exp_config_copy[key]]

    # Create the cartesian product
    exp_list = [
        dict(zip(exp_config_copy.keys(), values))
        for values in product(*exp_config_copy.values())
    ]

    return exp_list


def expand_dict_list(config_list):
    """Convenience function for expanding a list of experiment configurations.
    See `expand_config` for details on the expansion operation.
    :param config_list: list of configuration objects to expand.
    :returns: the list of experiment dictionaries.
    """

    def expand_plus(acc, value):
        return acc + expand_config(value)

    return reduce(expand_plus, config_list, initial=[])


def filter_dict_list(dict_list, keep=[], remove=[], filter_fn=None):
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

    keys_to_check = zip(keep + remove, [True] * len(keep) + [False] * len(remove))

    def key_filter(exp_dict):
        for entry, to_keep in keys_to_check:
            keys, values_to_check = entry
            values_to_check = as_list(values_to_check)
            keys = as_list(keys)

            exp_value = exp_dict
            for key in keys:
                exp_value = exp_dict[key]

            # keep the experiment
            return (exp_value in values_to_check and not to_keep) or (
                exp_value not in values_to_check and to_keep
            )

    if filter_fn is not None:

        def final_filter(exp_dict):
            return key_filter(exp_dict) or filter_fn(exp_dict)

    else:
        final_filter = key_filter

    return list(filter(final_filter, dict_list))


def hash_dict(exp_dict):
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
