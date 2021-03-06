"""
Command line helpers.
"""

from typing import Tuple, List, Optional
from argparse import ArgumentParser, Namespace
from warnings import warn

# default experiment arguments


def add_experiment_arguments(parser: Optional[ArgumentParser] = None) -> ArgumentParser:
    """Add default command line arguments for an experiment. By default, a new ArgumentParser is constructed and returned.
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
    # time experiment
    parser.add_argument(
        "-T",
        "--timed",
        dest="timed",
        action="store_true",
        help="Whether or not to time the experiment.",
    )
    # number of nodes on which to run.
    parser.add_argument(
        "-N",
        "--nodes",
        dest="nodes",
        type=int,
        default=None,
        help="Number of nodes on which to run the experiment.",
    )

    # specific experiment indices to run.
    parser.add_argument(
        "-I",
        "--indices",
        nargs="*",
        type=int,
        dest="indices",
        default=None,
        help="Indices of specific experiments to run.",
    )

    # sbatch file to use when submitting the job.
    parser.add_argument(
        "-B",
        "--sbatch",
        dest="sbatch",
        default=None,
        help="Script file to use with Slurm 'sbatch' command.",
    )

    # shuffle the data before running
    parser.add_argument(
        "--shuffle",
        dest="shuffle",
        action="store_true",
        help="Whether or not to shuffle the experiment order before running.",
    )

    # group experiments by dataset
    parser.add_argument(
        "--group_by_dataset",
        dest="group_by_dataset",
        action="store_true",
        help="Whether or not to group the experiments by dataset. Incompatible with --nodes and --shuffle.",
    )

    return parser


def add_plotting_arguments(parser: Optional[ArgumentParser] = None) -> ArgumentParser:
    """Add default command line arguments for plotting an experiment. By default, a new ArgumentParser is constructed and returned.
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
        nargs="*",
        required=True,
        dest="exp_id",
        help="The ID of the experiment to run.",
    )

    parser.add_argument(
        "-P",
        "--plot",
        nargs="*",
        required=True,
        dest="plot_name",
        help="The name of the plot to generate.",
    )

    # run in debug mode
    parser.add_argument(
        "--debug",
        action="store_true",
        dest="debug",
        help="Run in debug mode: exceptions end program execution and full stack-traces are printed.",
    )

    # location to save results
    parser.add_argument(
        "-R",
        "--results",
        dest="results_dir",
        default="results",
        help="Source directory for results.",
    )
    # location of source data
    parser.add_argument(
        "-F",
        "--figures",
        default="figures",
        dest="figures_dir",
        help="Destination directory for figures.",
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


def get_experiment_arguments(
    parser: Optional[ArgumentParser] = None,
) -> Tuple[
    Tuple[
        str,
        str,
        str,
        bool,
        bool,
        bool,
        bool,
        str,
        bool,
        int,
        List[int],
        str,
        bool,
        bool,
    ],
    Tuple[Namespace, List],
]:
    """Create and parse default experiment arguments from the command line. Default behavior is to create a new ArgumentParser object.
    :param parser: (Optional) an ArgumentParser instance to which the default arguments should be added.
    :returns: default arguments unpacked into a tuple, the parser, the arguments object, and an extra, unparsed arguments.
    """
    assert parser is None or isinstance(parser, ArgumentParser)

    # this will create the argument parser if necessary.
    parser = add_experiment_arguments(parser)
    arguments, extra = parser.parse_known_args()

    if len(extra) > 0:
        warn(f"Unknown command-line arguments {extra} encountered!")

    return unpack_experiment_defaults(arguments), (arguments, extra)


def get_plotting_arguments(
    parser: Optional[ArgumentParser] = None,
) -> Tuple[
    Tuple[List[str], List[str], str, str, bool, bool, str], Tuple[Namespace, List]
]:
    """Create and parse default plotting arguments from the command line. Default behavior is to create a new ArgumentParser object.
    :param parser: (Optional) an ArgumentParser instance to which the default arguments should be added.
    :returns: default arguments unpacked into a tuple, the parser, the arguments object, and an extra, unparsed arguments.
    """
    assert parser is None or isinstance(parser, ArgumentParser)

    # this will create the argument parser if necessary.
    parser = add_plotting_arguments(parser)
    arguments, extra = parser.parse_known_args()

    if len(extra) > 0:
        warn(f"Unknown command-line arguments {extra} encountered!")

    return unpack_plotting_defaults(arguments), (arguments, extra)


def unpack_experiment_defaults(
    arguments: Namespace,
) -> Tuple[
    str, str, str, bool, bool, bool, bool, str, bool, int, List[int], str, bool, bool
]:
    return (
        arguments.exp_id,
        arguments.data_dir,
        arguments.results_dir,
        arguments.force_rerun,
        arguments.save_results,
        arguments.verbose,
        arguments.debug,
        arguments.log_file,
        arguments.timed,
        arguments.nodes,
        arguments.indices,
        arguments.sbatch,
        arguments.shuffle,
        arguments.group_by_dataset,
    )


def unpack_plotting_defaults(
    arguments: Namespace,
) -> Tuple[List[str], List[str], str, str, bool, bool, str]:
    return (
        arguments.exp_id,
        arguments.plot_name,
        arguments.figures_dir,
        arguments.results_dir,
        arguments.verbose,
        arguments.debug,
        arguments.log_file,
    )


# convenience code for testing default command-line arguments.
if __name__ == "__main__":
    print(get_experiment_arguments())
