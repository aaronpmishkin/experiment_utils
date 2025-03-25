"""
Predefine plotting functions for use with plot_grid.py
"""

from __future__ import annotations
from typing import Callable, Any
from copy import deepcopy

import numpy as np
import torch
import matplotlib.pyplot as plt  # type: ignore

bandwidth = 0.05


def make_convergence_plot(
    ax: plt.Axes,
    results: dict[str, np.ndarray],
    line_kwargs: dict,
    settings: dict,
    error_bars: bool = True,
) -> None:
    """Generate convergence plot with optional error bars.

    Params:
        ax: axis object.
        results: dict or dict of dicts. The first level is indexed by 'line'
            keys, the second level contains either (a) an np.array or list of
            metric values to plot, or (b) summary statistics indexed by
            'center', 'upper', and 'lower'.
        line_kwargs: dict of key-work arguments for each bar.
        settings: configuration object for plot. See `plotting.defaults`.
        error_bars: (optional) whether or not to plot error bars. If `True`,
            each results[line] dictionary must have keys `'center'`, `'upper'`,
            and `'lower'`.
    """
    lines = results.keys()
    plt_lines = {}

    for line in lines:
        plt_lines[line] = {}
        if error_bars:
            assert (
                "center" in results[line]
                and "upper" in results[line]
                and "lower" in results[line]
            )
            y = results[line]["center"]

            if "x" in results[line]:
                x = results[line]["x"] + 1
                if len(y) >= len(x):
                    y = y[0 : len(x)]
                else:
                    x = x[0 : len(y)]
            else:
                x = np.arange(len(y))

            plt_lines[line]["fill"] = ax.fill_between(
                x,
                results[line]["lower"][0 : len(x)],
                results[line]["upper"][0 : len(x)],
                alpha=settings["error_alpha"],
                color=line_kwargs[line]["c"],
            )
        else:
            y = results[line]["center"]
            x = np.arange(len(y))

        plt_lines[line]["line"] = ax.plot(
            x, y, alpha=settings["line_alpha"], **line_kwargs[line]
        )
        if settings.get("star_final", False):
            plt_lines[line]["star"] = ax.plot(
                [x[-1]],
                [y[-1]],
                color=line_kwargs[line]["c"],
                marker="*",
                markersize=22,
            )


def update_convergence_plot(
    ax: plt.Axes,
    results: dict[str, tuple[Callable, Callable]],
    plt_lines: dict[Any, list[plt.Line2D]],
):

    lines = list(results.keys())
    artists = []
    for line in lines:
        y = results[line]["center"]
        x = plt_lines[line]["line"][0].get_xdata()

        if "fill" in plt_lines:
            # make dummy fill
            dummy_fill = ax.fill_between(
                x,
                results[line]["lower"][0 : len(x)],
                results[line]["upper"][0 : len(x)],
                alpha=0,
            )
            # extract plot data from dummy fill
            paths = dummy_fill.get_paths()[0]
            # remove dummy fill from plot
            dummy_fill.remove()
            # update fill data
            plt_lines[line]["fill"].set_paths([paths.vertices])
            artists.append(plt_lines[line]["fill"])

        if "line" in plt_lines:
            plt_lines[line]["line"][0].set_ydata(y)
            artists.append(plt_lines[line]["line"][0])

        if "star" in plt_lines:
            plt_lines[line]["star"][0].set_ydata([y[-1]])
            artists.append(plt_lines[line]["star"][0])

        ax.autoscale(enable=True, axis="y", tight=True)

        return artists


def make_error_bar_plot(
    ax: plt.Axes,
    results: dict[str, np.ndarray],
    line_kwargs: dict,
    settings: dict,
    error_bars: bool = True,
) -> None:
    """Generate error-bar plot of final metrics.

    Params:
        ax: axis object.
        results: dict or dict of dicts. The first level is indexed by 'line'
            keys, the second level contains either (a) an np.array or list of
            metric values to plot, or (b) summary statistics indexed by
            'center', 'upper', and 'lower'.
        line_kwargs: dict of key-work arguments for each bar.
        settings: configuration object for plot. See `plotting.defaults`.
        error_bars: (optional) whether or not to plot error bars. If `True`,
            each results[line] dictionary must have keys `'center'`, `'upper'`,
            and `'lower'`.
    """
    lines = results.keys()

    for line in lines:
        if "center" not in results[line]:
            continue

        y = np.abs(results[line]["center"])

        if "x" in results[line]:
            x = results[line]["x"]
        else:
            x = np.arange(len(y)) + 1

        ax.scatter(
            x=x,
            y=y,
            # yerr=[
            #     results[line]["center"] - results[line]["lower"],
            #     results[line]["upper"] - results[line]["center"],
            # ],
            # capsize=10,
            c=line_kwargs[line]["c"],
            label=line_kwargs[line]["label"],
        )
