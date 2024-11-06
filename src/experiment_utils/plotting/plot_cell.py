"""
Predefine plotting functions for use with plot_grid.py
"""

from __future__ import annotations
from typing import Callable, Any

import numpy as np
import torch
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


def plot_model_fits(
    ax: plt.Axes,
    results: dict[str, tuple[Callable, Callable]],
    line_kwargs: dict,
    settings: dict,
    error_bars: bool = True,
):

    lines = list(results.keys())
    # choose first repeat to plot.
    load_fn = results[lines[0]]
    _, (x, y) = load_fn()
    indices = np.argsort(x.squeeze().numpy())
    x = x[indices]
    y = y[indices]

    # make scatter plot of data
    ax.scatter(x, y, color="k", s=100)

    for line in lines:
        model, _ = results[line]()
        model_fit = model(x).detach().numpy()
        ax.plot(x, model_fit, alpha=settings["line_alpha"], **line_kwargs[line])


def plot_neuron_norms(
    ax: plt.Axes,
    results: dict[str, tuple[Callable, Callable]],
    line_kwargs: dict,
    settings: dict,
    error_bars: bool = True,
):

    lines = list(results.keys())

    for line in lines:
        model, _ = results[line]()

        norms = model.get_neuron_norms().numpy()
        # compute neuron norms
        norms = np.flip(np.sort(norms))
        # keep only top 100
        norms = norms[0:100]

        x = np.arange(len(norms))

        ax.plot(x, norms, alpha=settings["line_alpha"], **line_kwargs[line])


def plot_breakpoints(
    ax: plt.Axes,
    results: dict[str, tuple[Callable, Callable]],
    line_kwargs: dict,
    settings: dict,
    error_bars: bool = True,
):

    lines = list(results.keys())

    for line in lines:
        model, (x, _) = results[line]()
        x = np.sort(x.squeeze())

        breakpoints = model.get_breakpoints().detach().numpy()

        plt.hist(
            breakpoints,
            bins=x,
            color=line_kwargs[line]["c"],
            label=line_kwargs[line]["label"],
        )


def plot_neuron_fits(
    ax: plt.Axes,
    results: dict[str, tuple[Callable, Callable]],
    line_kwargs: dict,
    settings: dict,
    error_bars: bool = True,
):

    lines = list(results.keys())
    # choose first repeat to plot.
    load_fn = results[lines[0]]
    _, (x, y) = load_fn()
    indices = np.argsort(x.squeeze().numpy())
    x = x[indices]
    y = y[indices]

    x_min = x.min().numpy()
    x_max = x.max().numpy()

    data_range = np.linspace(2 * x_min, 2 * x_max, num=1000).reshape(-1, 1)

    # make scatter plot of data
    ax.scatter(x, y, color="k", s=25)
    ax.axhline(y=0.0, color="k", linestyle="-", linewidth=1.5)
    eps = 0.1

    ax.set_xlim(x_min - eps, x_max + eps)
    y_max = y.max().numpy()
    ax.set_ylim(-y_max - eps, y_max + eps)
    plt_lines = {}

    for line in lines:
        model, _ = results[line]()
        neuron_fits = model.neuron_fits(data_range).detach().numpy()
        plt_lines[line] = ax.plot(
            data_range.squeeze(),
            neuron_fits,
            alpha=settings["line_alpha"],
            **line_kwargs[line]["neurons"],
        )
        breakpoints = model.get_breakpoints().detach().numpy()
        # ax.scatter(
        #     breakpoints,
        #     np.zeros_like(breakpoints),
        #     color="k",
        #     s=100,
        #     marker="*",
        # )

        model_fit = (
            model(torch.tensor(data_range, dtype=torch.get_default_dtype()))
            .detach()
            .numpy()
        )
        model_line = ax.plot(
            data_range.squeeze(),
            model_fit,
            alpha=settings["line_alpha"],
            **line_kwargs[line]["model"],
        )
        plt_lines[line].append(model_line[0])

    return plt_lines


def update_neuron_fits(
    ax: plt.Axes,
    results: dict[str, tuple[Callable, Callable]],
    plt_lines: dict[Any, list[plt.Line2D]],
):

    lines = list(results.keys())
    artists = []
    for line in lines:
        model, _ = results[line]()
        data_range = plt_lines[line][0].get_xdata().reshape(-1, 1)
        neuron_fits = model.neuron_fits(data_range).detach().numpy()
        model_fit = (
            model(torch.tensor(data_range, dtype=torch.get_default_dtype()))
            .detach()
            .numpy()
        )

        for i, line_obj in enumerate(plt_lines[line]):
            if i == neuron_fits.shape[-1]:
                line_obj.set_ydata(model_fit)
            else:
                line_obj.set_ydata(neuron_fits[:, i])

            artists.append(line_obj)

        return artists
