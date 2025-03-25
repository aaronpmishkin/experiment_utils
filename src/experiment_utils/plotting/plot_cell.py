"""
Predefine plotting functions for use with plot_grid.py
"""

from __future__ import annotations
from typing import Callable, Any
from copy import deepcopy

import numpy as np
import torch
import matplotlib.pyplot as plt  # type: ignore
from scipy.stats import gaussian_kde

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

    for line in lines:
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
    (x, y) = load_fn()[1]
    indices = np.argsort(x.squeeze().numpy())
    x = x[indices]
    y = y[indices]

    # make scatter plot of data
    ax.scatter(x, y, color="k", s=100)

    for line in lines:
        model = results[line]()[0]
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
        model = results[line]()[0]

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
        out = results[line]()[0]
        model = out[0]
        (x, _) = out[1]

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
    (x, y) = load_fn()[1]
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
        model = results[line]()[0]
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
        model = results[line]()[0]
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


def plot_fit_on_ball(
    ax: plt.Axes,
    results: dict[str, tuple[Callable, Callable]],
    line_kwargs: dict,
    settings: dict,
    error_bars: bool = True,
):
    lines = list(results.keys())
    # choose first repeat to plot.
    load_fn = results[lines[0]]
    (x, y) = load_fn()[1]
    if torch.is_tensor(x):
        x = x.squeeze().numpy()
    if torch.is_tensor(y):
        y = y.squeeze().numpy()

    # compute polar coordinates
    theta = compute_normalized_polar_angle(x)

    indices = np.argsort(theta)
    theta = theta[indices]
    y = y[indices]

    # make scatter plot of data
    ax.scatter(theta, y, color="k", s=25)
    ax.axhline(y=0.0, color="k", linestyle="-", linewidth=1.5)

    theta_range = np.linspace(0, 2 * np.pi, 1000)
    norm_theta_range = theta_range / (2 * np.pi)
    x = np.cos(theta_range)
    y = np.sin(theta_range)
    data_range = np.stack([x, y]).T

    pt_data_range = torch.tensor(data_range, dtype=torch.get_default_dtype())
    plt_lines = {}

    for line in lines:
        model = results[line]()[0]
        model_fit = model.forward(pt_data_range).detach().numpy()

        if np.any(np.isnan(model_fit)):
            continue

        plt_lines[line] = ax.plot(
            norm_theta_range,
            model_fit,
            alpha=settings["line_alpha"],
            **line_kwargs[line],
        )

    return plt_lines


def update_fit_on_ball(
    ax: plt.Axes,
    results: dict[str, tuple[Callable, Callable]],
    plt_lines: dict[Any, list[plt.Line2D]],
):

    lines = list(results.keys())
    artists = []

    theta_range = np.linspace(0, 2 * np.pi, 1000)
    x_1 = np.cos(theta_range)
    x_2 = np.sin(theta_range)
    data_range = np.stack([x_1, x_2]).T

    pt_data_range = torch.tensor(data_range, dtype=torch.get_default_dtype())

    for line in lines:
        model = results[line]()[0]
        model_fit = model.forward(pt_data_range).detach().numpy()
        line_obj = plt_lines[line][0]
        line_obj.set_ydata(model_fit)
        artists.append(line_obj)

    return artists


def plot_neuron_density(
    ax: plt.Axes,
    results: dict[str, tuple[Callable, Callable]],
    line_kwargs: dict,
    settings: dict,
    error_bars: bool = True,
    bandwidth=bandwidth,
):

    lines = list(results.keys())
    plt_lines = {}
    radian_interval = np.linspace(0, 1, 200)
    load_fn = results[lines[0]]
    # z = load_fn()[2]

    # z = np.stack([z, -z])
    # theta_star = compute_normalized_polar_angle(z)
    # ax.scatter(
    #     theta_star,
    #     np.zeros_like(theta_star),
    #     color="y",
    #     marker="*",
    #     s=200,
    # )

    for line in lines:
        plt_lines[line] = []
        model = results[line]()[0]
        theta = neuron_angles(model)
        weights = np.squeeze(model.final_layer.weight.detach().numpy())
        line_kwargs[line]["linestyle"] = "None"
        line_kwargs[line]["markevery"] = 1
        line_kwargs[line]["markersize"] = 0.5

        theta_pos = theta[weights >= 0]
        weights_pos = weights[weights >= 0]
        # kde_pos = gaussian_kde(theta_pos, weights=weights_pos, bw_method=bandwidth)
        # density_pos = kde_pos(radian_interval)

        plt_lines[line] += ax.plot(
            theta_pos,
            weights_pos,
            alpha=settings["line_alpha"],
            **line_kwargs[line],
        )

        theta_neg = theta[weights < 0]
        weights_neg = weights[weights < 0]
        # kde_neg = gaussian_kde(theta_neg, weights=-1*weights_neg, bw_method=bandwidth)
        # density_neg = kde_neg(radian_interval)

        neg_kwargs = deepcopy(line_kwargs[line])
        neg_kwargs["c"] = "red"
        neg_kwargs["label"] = ""

        plt_lines[line] += ax.plot(
            theta_neg,
            weights_neg,
            alpha=settings["line_alpha"],
            **neg_kwargs,
        )

        net_weights = weights_pos + weights_neg
        # kde_net = gaussian_kde(theta_pos, weights=net_weights, bw_method=bandwidth)
        # net_density = kde_net(radian_interval)
        net_kwargs = deepcopy(line_kwargs[line])
        net_kwargs["c"] = "black"

        plt_lines[line] += ax.plot(
            theta_pos,
            net_weights,
            alpha=settings["line_alpha"],
            **net_kwargs,
        )

    return plt_lines


def update_neuron_density(
    ax: plt.Axes,
    results: dict[str, tuple[Callable, Callable]],
    plt_lines: dict[Any, list[plt.Line2D]],
    bandwidth=bandwidth,
):

    line_names = list(results.keys())
    radian_interval = np.linspace(0, 1, 200)
    artists = []

    densities = []
    for line in line_names:
        model = results[line]()[0]
        theta = neuron_angles(model)
        weights = np.squeeze(model.final_layer.weight.detach().numpy())

        theta_pos = theta[weights >= 0]
        weights_pos = weights[weights >= 0]
        # kde_pos = gaussian_kde(theta_pos, weights=weights_pos, bw_method=bandwidth)
        # density_pos = kde_pos(radian_interval)

        plt_lines[line][0].set_ydata(weights_pos)
        # densities += density_pos.tolist()

        theta_neg = theta[weights < 0]
        weights_neg = weights[weights < 0]
        # kde_neg = gaussian_kde(theta_neg, weights=-1*weights_neg, bw_method=bandwidth)
        # density_neg = kde_neg(radian_interval)
        plt_lines[line][1].set_ydata(weights_neg)
        # densities += density_neg.tolist()

        net_weights = weights_pos + weights_neg
        # kde_net = gaussian_kde(theta_pos, weights=net_weights, bw_method=bandwidth)
        # net_density = kde_net(radian_interval)
        plt_lines[line][2].set_ydata(net_weights)
        artists = [plt_lines[line][0], plt_lines[line][1], plt_lines[line][2]]

        # densities += net_density.tolist()

    ax.margins(0.05, 0.05)
    ax.set_ylim(1.05 * min(weights_neg), 1.05 * max(weights_pos))

    return artists


def plot_finite_differences(
    ax: plt.Axes,
    results: dict[str, tuple[Callable, Callable]],
    line_kwargs: dict,
    settings: dict,
    error_bars: bool = True,
    bandwidth=bandwidth,
):

    lines = list(results.keys())
    plt_lines = {}

    for line in lines:
        plt_lines[line] = []
        model = results[line]()[0]
        theta = neuron_angles(model)
        weights = np.squeeze(model.final_layer.weight.detach().numpy())

        line_kwargs[line]["linestyle"] = "None"
        line_kwargs[line]["markevery"] = 1
        line_kwargs[line]["markersize"] = 0.5

        theta_pos = theta[weights >= 0]
        weights_pos = weights[weights >= 0]
        steps = theta_pos[1:] - theta_pos[:-1]
        diffs_pos = (weights_pos[1:] - weights_pos[:-1]) / steps

        plt_lines[line] += ax.plot(
            theta_pos[:-1],
            diffs_pos,
            alpha=settings["line_alpha"],
            **line_kwargs[line],
        )

        theta_neg = theta[weights < 0]
        weights_neg = weights[weights < 0]
        steps = theta_neg[1:] - theta_neg[:-1]
        diffs_neg = (weights_neg[1:] - weights_neg[:-1]) / steps

        neg_kwargs = deepcopy(line_kwargs[line])
        neg_kwargs["c"] = "red"
        neg_kwargs["label"] = ""

        plt_lines[line] += ax.plot(
            theta_pos[:-1],
            diffs_neg,
            alpha=settings["line_alpha"],
            **neg_kwargs,
        )

    return plt_lines


def update_finite_differences(
    ax: plt.Axes,
    results: dict[str, tuple[Callable, Callable]],
    plt_lines: dict[Any, list[plt.Line2D]],
    bandwidth=bandwidth,
):

    line_names = list(results.keys())
    radian_interval = np.linspace(0, 1, 200)
    artists = []

    densities = []
    for line in line_names:
        model = results[line]()[0]
        theta = neuron_angles(model)
        weights = np.squeeze(model.final_layer.weight.detach().numpy())

        theta_pos = theta[weights >= 0]
        weights_pos = weights[weights >= 0]
        steps = theta_pos[1:] - theta_pos[:-1]
        diffs_pos = (weights_pos[1:] - weights_pos[:-1]) / steps

        plt_lines[line][0].set_ydata(diffs_pos)

        theta_neg = theta[weights < 0]
        weights_neg = weights[weights < 0]
        steps = theta_neg[1:] - theta_neg[:-1]
        diffs_neg = (weights_neg[1:] - weights_neg[:-1]) / steps
        plt_lines[line][1].set_ydata(diffs_neg)

        artists = [plt_lines[line][0], plt_lines[line][1]]

        # densities += net_density.tolist()

    ax.margins(0.05, 0.05)
    ax.set_ylim(1.05 * min(diffs_neg), 1.05 * max(diffs_pos))

    return artists


def compute_normalized_polar_angle(x):
    # points must be on the plane
    assert len(x.shape) == 2
    _, d = x.shape
    assert d == 2

    # convert x into polar coordinates
    theta = np.arctan2(x[:, 1], x[:, 0])
    # normalize coordinates
    theta = theta / (2 * np.pi)
    # remove negative coordinates
    theta[theta < 0] = theta[theta < 0] + 1

    return theta


def neuron_angles(model):
    # one hidden layer with neurons on the sphere
    assert model.input_size == 2
    assert len(model.hidden_layers) == 1

    # compute signed angle
    input_weights = model.hidden_layers[0].weight.detach().numpy()
    theta = compute_normalized_polar_angle(input_weights)

    return theta


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)
