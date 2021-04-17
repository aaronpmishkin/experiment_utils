"""
Command line helpers.
"""

from typing import Tuple, List, Optional
from argparse import ArgumentParser, Namespace
from warnings import warn


def add_default_arguments(parser: Optional[ArgumentParser] = None) -> ArgumentParser:
    """Add default command line arguments for an experiment. By default, a new ArgmentParser is constructed and returned.
    :param parser: (optional) parser to which the default arguments should be added.
    :returns: parser with default arguments added.
    """
    assert parser is None or isinstance(parser, ArgumentParser)

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
    # run in debug mode
    parser.add_argument(
        "--debug",
        action="store_true",
        dest="debug",
        help="Run in debug mode: exceptions end program execution and full stack-traces are printed.",
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
    # run verbosely.
    parser.add_argument(
        "-V",
        "--verbose",
        action="store_true",
        dest="verbose",
        help="Print logging information verbosely.",
    )
    # log-file
    parser.add_argument(
        "-L",
        "--log",
        dest="log_file",
        default=None,
        help="Path to log-file for logger output.",
    )

    return parser


def get_default_arguments(
    parser: Optional[ArgumentParser] = None,
) -> Tuple[Tuple[str, str, str, bool, bool, bool, bool, str], Tuple[Namespace, List]]:
    """Create and parse default experiment arguments from the command line. Default behavior is to create a new ArgumentParser object.
    :param parser: (Optional) an ArgumentParser instance to which the default arguments should be added.
    :returns: default arguments unpacked into a tuple, the parser, the arguments object, and an extra, unparsed arguments.
    """
    assert parser is None or isinstance(parser, ArgumentParser)

    # this will create the argument parser if necessary.
    parser = add_default_arguments(parser)
    arguments, extra = parser.parse_known_args()
    print(type(arguments), type(extra))

    if len(extra) > 0:
        warn(f"Unknown command-line arguments {extra} encountered!")

    return unpack_defaults(arguments), (arguments, extra)


def unpack_defaults(
    arguments: Namespace,
) -> Tuple[str, str, str, bool, bool, bool, bool, str]:
    return (
        arguments.exp_id,
        arguments.data_dir,
        arguments.results_dir,
        arguments.force_rerun,
        arguments.save_results,
        arguments.verbose,
        arguments.debug,
        arguments.log_file,
    )


# convenience code for testing default command-line arguments.
if __name__ == "__main__":
    print(get_default_arguments())
