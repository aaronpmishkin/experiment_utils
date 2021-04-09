"""
Command line helpers.
"""

from argparse import ArgumentParser
from warnings import warn


def add_default_arguments(parser=None):
    """Add default command line arguments for an experiment. By default, a new ArgmentParser is constructed and returned.
    :param parser: (optional) argparse.ArgumentParser to which the default arguments should be added.
    :returns: ArgumentParser with default arguments added.
    """
    assert (parser is None or isinstance(parser, ArgumentParser))

    # create argument parser.
    if parser is None:
        parser = ArgumentParser()

    # default arguments #

    # experiment id
    parser.add_argument(
        "-E",
        "--experiment",
        required=True,
        dest="exp_id",
        help="The ID of the experiment to run.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        dest="debug",
        help="The ID of the experiment to run.",
    )

    # force re-run of experiment
    parser.add_argument(
        "-F",
        "--force",
        action="store_true",
        dest="force_rerun",
        help="Force re-run of the experiment.",
    )
    # save results
    parser.add_argument(
        "-S",
        "--save",
        default=1,
        type=int,
        dest="save_results",
        help="Save results of the experiment. Defaults to True.",
    )
    # location to save results
    parser.add_argument(
        "-R",
        "--results",
        dest="results_dir",
        default="results",
        help="Destination directory for results.",
    )
    # location of source data
    parser.add_argument(
        "-D",
        "--data",
        default="data",
        dest="data_dir",
        help="Source directory for data files.",
    )
    # location of source data
    parser.add_argument(
        "-V",
        "--verbose",
        action="store_true",
        dest="verbose",
        help="Print logging information verbosely.",
    )

    return parser


def get_default_arguments(parser=None):
    """Create and parse default experiment arguments from the command line. Default behavior is to create a new ArgumentParser object.
    :param parser: (Optional) an ArgumentParser instance to which the default arguments should be added.
    :returns: default arguments unpacked into a tuple, the parser, the arguments object, and an extra, unparsed arguments.
    """
    assert (parser is None or isinstance(parser, ArgumentParser))

    parser = add_default_arguments(parser)
    arguments, extra = parser.parse_known_args()

    if len(extra) > 0:
        warn(f"Unknown command-line arguments {extra} encountered!")

    return unpack_defaults(arguments), (arguments, extra)


def unpack_defaults(arguments):
    return arguments.exp_id, arguments.data_dir, arguments.results_dir, arguments.force_rerun, arguments.save_results, arguments.verbose, arguments.debug


# convenience code for testing default command-line arguments.
if __name__ == "__main__":
    print(get_default_arguments())
