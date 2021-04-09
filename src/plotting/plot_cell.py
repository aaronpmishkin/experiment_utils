"""
Predefine plotting functions for use with plot_grid.py
"""

import numpy as np
import matplotlib.pyplot as plt


def make_convergence_plot(ax, results, line_kwargs, settings, error_bars=True):
    """Generate convergence plot with optional error bars for methods indexed by `lines`.
    :param ax: axis object.
    :param results: dict or dict of dicts. The first level is indexed by 'line' keys.
        The second level contains either (a) an np.array or list of loss values to plot, or (b)
        summary statistics indexed by 'center', 'upper', and 'lower'.
    :param line_kwargs: dict of key-work arguments for each bar.
    :param settings: configuration object for plot. See DEFAULT_SETTINGS above.
    :param error_bars: (optional) whether or not to plot error bars. If 'True', each results[line] dictionary must have keys "center", "upper", and "lower".
    """
    lines = results.keys()

    for i, line in enumerate(lines):
        if error_bars:
            assert "center" in results[line] and "upper" in results[line] and "lower" in results[line]
            y = results[line]["center"]
            x = np.arange(len(y))

            ax.fill_between(
                x,
                results[line]["lower"],
                results[line]["upper"],
                alpha=settings["error_alpha"],
                color=line_kwargs[line]["c"],
            )
        else:
            y = results[line]["center"]
            x = np.arange(len(y))

        ax.plot(x, y, alpha=settings["line_alpha"], **line_kwargs[line])