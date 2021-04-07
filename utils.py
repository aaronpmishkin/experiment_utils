"""
General utilities
"""


def as_list(x):
    """ Wrap argument into a list if it is not iterable.
        :param x: a (potential) singleton to wrap in a list.
        :returns: [x] if x is not iterable and x if it is.
    """
    try:
        iterator = iter(x)
        return x
    except TypeError:
        return list(x)
