"""
Utility for generating grids of plots.
"""

import os
import math
import itertools
from typing import Any, Callable, Optional

import matplotlib.pyplot as plt  # type: ignore

from experiment_utils.plotting.defaults import DEFAULT_SETTINGS


def try_cell_row_col(
    value_dict: dict, row: Any, col: Any, default_value: Optional[bool] = None
) -> Any:
    """Helper for finding the dictionary value associated with a cell in the plot.
    The dictionary is first index by cell using `(row, col)`, then by row using `row`,
    and finally by column using `col`. `default_value` is returned if nothing is found.
    :param value_dict: dictionary to index into. e.g y-axis labels for each cell.
    :param row: key for the plot row.
    :param col: key for the plot column.
    :param default_value: value to return if nothing is found. Optional. Defaults to `None`.
    :returns: value_dict[(row, col)], value_dict[row], value_dict[col] or `default_value`.
    """
    return value_dict.get(
        (row, col), value_dict.get(row, value_dict.get(col, default_value))
    )


def plot_grid(
    plot_fn: Callable,
    results: dict,
    figure_labels: dict,
    line_kwargs: dict,
    limits: dict = {},
    ticks: dict = {},
    log_scale: dict = {},
    settings: dict = DEFAULT_SETTINGS,
    base_dir: Optional[str] = None,
):
    """Helper function for generating a len(rows) x len(cols) grid of plots. In the following, cell refers to a (row, col) key-pair.
    :param plot_fn: function for plotting each cell in the grid.
    :param results: nested dictionary of results. The first level of is defines the rows of the plot,
        the second the columns and the third  the lines in each cell. Note that the number of columns must be
        the same for each row, but the number of lines may differ for each cell.
    :param figure_labels: dict of dicts containing labels for the plots. The top-level dict should contain keys
        'y_labels', 'x_labels', 'col_titles', 'row_titles'. Each sub-dict can be indexed by cell, row, or column.
    :param line_kwargs: dict of key-word arguments for each key in 'lines'.
    :param limits: dict of tuples (x_lim, y_lim), where x_lim are the desired limits for the x-axis and
        y_lim are the desired limits for the y-axis. Can be indexed by cell, row, or column.
    :param ticks: dict of tuples (x_ticks, y_ticks), where x_ticks are the desiredd ticks for the x-axis
        y_ticks are the desired ticks for the y-axis. Can be indexed by cell, row, or column.
    :param log_scale: dict of strings indicating whether or not to plot a cell, row, or column as a log-log,
        log-linear, or linear-linear plot. Defaults to linear-linear.
    :param settings: dict with the plot configuration. See 'defaults.DEFAULT_SETTINGS' above.
    :param base_dir: location to save plot. Defaults to 'None', in which case the plot is not saved.
    :returns: figure and dictionary of axis objects indexed by the rows and columns.
    """
    rows = list(results.keys())
    cols = list(results[rows[0]].keys())

    fig = plt.figure(
        figsize=(settings["fig_width"] * len(cols), len(rows) * settings["fig_height"])
    )

    # grid spec.
    spec = fig.add_gridspec(ncols=len(cols), nrows=len(rows))

    # title the plot if a title is given.
    if "title" in figure_labels:
        fig.suptitle(
            figure_labels["title"], fontsize=settings.get("titles_fs", 18), y=1
        )

    # unpack label arguments:
    y_labels, x_labels, col_titles, row_titles = (
        figure_labels.get("y_labels", {}),
        figure_labels.get("x_labels", {}),
        figure_labels.get("col_titles", {}),
        figure_labels.get("row_titles", {}),
    )
    axes = {}

    for i, (row, col) in enumerate(itertools.product(rows, cols)):
        ax = fig.add_subplot(spec[math.floor(i / len(cols)), i % len(cols)])
        # dict of axes objects
        axes[(row, col)] = ax
        ax.yaxis.offsetText.set_fontsize(settings["offest_text_fs"])

        # in the top row
        if settings.get("col_titles", False) and i < len(cols):
            ax.set_title(col_titles.get(col, ""), fontsize=settings["subtitle_fs"])

        if settings.get("row_titles", False) and i % len(cols) == 0:
            ax.annotate(
                row_titles.get(row, ""),
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - settings["row_title_pad"], 0),
                xycoords=ax.yaxis.label,
                textcoords="offset points",
                fontsize=settings["subtitle_fs"],
                ha="right",
                va="center",
                rotation=90,
            )

        # start of a new row
        if settings.get("y_labels", False) == "left_col" and i % len(cols) == 0:
            ax.set_ylabel(y_labels.get(row, ""), fontsize=settings["axis_labels_fs"])
        elif settings.get("y_labels", False) == "every_col":
            ax.set_ylabel(
                try_cell_row_col(y_labels, row, col, ""),
                fontsize=settings["axis_labels_fs"],
            )

        # in the bottom row
        if (
            settings.get("x_labels", False) == "bottom_row"
            and len(cols) * (len(rows) - 1) <= i
        ):
            ax.set_xlabel(x_labels.get(col, ""), fontsize=settings["axis_labels_fs"])
        elif settings.get("x_labels", False) == "every_row":
            ax.set_xlabel(
                try_cell_row_col(x_labels, row, col, ""),
                fontsize=settings["axis_labels_fs"],
            )

        ax.ticklabel_format(
            axis="y",
            style=settings.get("ticklabel_format", "scientific"),
            scilimits=(0, 0),
        )

        # ticks
        ax.tick_params(labelsize=settings["tick_fs"])
        if try_cell_row_col(ticks, row, col, None) is not None:

            x_ticks, y_ticks = try_cell_row_col(ticks, row, col, None)

            if x_ticks is not None and len(x_ticks) > 0:
                ax.xticks(x_ticks)

            if y_ticks is not None and len(y_ticks) > 0:
                ax.yticks(y_ticks)

        # plot the cell
        plot_fn(ax, results[row][col], line_kwargs, settings)

        # log-scale:
        if try_cell_row_col(log_scale, row, col, None) is not None:
            log_type = try_cell_row_col(log_scale, row, col, None)
            if log_type == "log-linear":
                ax.set_yscale("log")
            elif log_type == "log-log":
                ax.set_yscale("log")
                ax.set_xscale("log")

        # limits: needs to be done after plotting the data
        if try_cell_row_col(limits, row, col, None) is not None:

            x_limits, y_limits = try_cell_row_col(limits, row, col, None)

            if x_limits is not None and len(x_limits) > 0:
                ax.set_xlim(*x_limits)

            if y_limits is not None and len(y_limits) > 0:
                ax.set_ylim(*y_limits)

    # Put only one shared legend to avoid clutter
    handles, labels = ax.get_legend_handles_labels()
    final_handles, final_labels = [], []
    for i, label in enumerate(labels):
        final_handles.append(handles[i])
        final_labels.append(labels[i])

    ncol = settings["legend_cols"]

    legend = fig.legend(
        final_handles,
        final_labels,
        loc="lower center",
        borderaxespad=0.1,
        fancybox=False,
        shadow=False,
        ncol=ncol,
        fontsize=settings["legend_fs"],
    )

    for line in legend.get_lines():
        line.set_linewidth(settings.get("legend_lw", 2.0))

    bottom_margin = settings["bottom_margin"] / len(rows)

    fig.subplots_adjust(
        wspace=settings.get("wspace", 0.2),
        hspace=settings.get("vspace", 0.2),
        bottom=bottom_margin,
    )

    if base_dir is not None:
        head, _ = os.path.split(base_dir)
        os.makedirs(head, exist_ok=True)
        plt.savefig(base_dir)

    return fig, axes
