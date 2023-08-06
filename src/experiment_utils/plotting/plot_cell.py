"""
Predefine plotting functions for use with plot_grid.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt  # type: ignore


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

    for line in lines:
        if error_bars:
            assert (
                "center" in results[line]
                and "upper" in results[line]
                and "lower" in results[line]
            )
            y = results[line]["center"]

            if "x" in results[line]:
                x = results[line]["x"]
                y = y[0 : len(x)]
            else:
                x = np.arange(len(y))

            ax.fill_between(
                x,
                results[line]["lower"][0 : len(x)],
                results[line]["upper"][0 : len(x)],
                alpha=settings["error_alpha"],
                color=line_kwargs[line]["c"],
            )
        else:
            y = results[line]["center"]
            x = np.arange(len(y))

        ax.plot(x, y, alpha=settings["line_alpha"], **line_kwargs[line])
        if settings.get("star_final", False):
            ax.plot(
                [x[-1]],
                [y[-1]],
                color=line_kwargs[line]["c"],
                marker="*",
                markersize=22,
            )


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
        assert (
            "center" in results[line]
            and "upper" in results[line]
            and "lower" in results[line]
        )
        y = results[line]["center"]

        if "x" in results[line]:
            x = results[line]["x"]
        else:
            x = np.arange(len(y)) + 1

        y = results[line]["center"]
        ax.errorbar(
            x=x,
            y=y,
            yerr=[
                results[line]["center"] - results[line]["lower"],
                results[line]["upper"] - results[line]["center"],
            ],
            capsize=10,
            **line_kwargs[line],
        )
