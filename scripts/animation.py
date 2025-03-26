"""
Plot an experiment.
"""

import os
from typing import List, Dict
from functools import reduce

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from experiment_utils import configs, utils, files, command_line
from experiment_utils.plotting import plot_grid, plot_cell, defaults

# merge experiment dictionaries.
from exp_configs import EXPERIMENTS
from plot_configs import PLOTS


INTERVAL_LENGTH = 25

# Script #

if __name__ == "__main__":
    (
        (
            exp_ids,
            plot_names,
            figures_dir,
            results_dir_base,
            verbose,
            debug,
            log_file,
        ),
        _,
        _,
    ) = command_line.get_plotting_arguments()

    logger_name = reduce(lambda acc, x: f"{acc}{x}_", exp_ids, "")
    logger = utils.get_logger(logger_name, verbose, debug, log_file)

    # lookup experiment #
    for exp_id in exp_ids:
        if exp_id not in EXPERIMENTS:
            raise ValueError(f"Experiment id {exp_id} is not in the experiment list!")

    config_list: List[Dict] = reduce(
        lambda acc, eid: acc + EXPERIMENTS[eid], exp_ids, []
    )
    results_dir = [os.path.join(results_dir_base, eid) for eid in exp_ids]

    logger.info(f"\n\n====== Making {plot_names} from {exp_ids} results ======\n")

    for plot_name in plot_names:
        dest = os.path.join(figures_dir, "_".join(exp_ids))
        plot_config_list = PLOTS[plot_name]

        for i, plot_config in enumerate(plot_config_list):
            logger.info(f"\n\n Creating {i+1}/{len(plot_config_list)} \n")

            parameterized_grid = files.load_parameterized_experiments(
                config_list,
                results_dir,
                metrics=plot_config["metrics"],
                row_key=plot_config["row_key"],
                line_key=plot_config["line_key"],
                repeat_key=plot_config["repeat_key"],
                variation_key=plot_config["variation_key"],
                target_metric=plot_config.get("target_metric", None),
                maximize_target=plot_config.get("maximize_target", False),
                metric_fn=plot_config["metrics_fn"],
                keep=plot_config.get("keep", []),
                remove=plot_config.get("remove", []),
                filter_fn=plot_config.get("filter_fn", None),
                transform_fn=plot_config.get("transform_fn", None),
                processing_fns=plot_config.get("processing_fns", None),
                x_key=plot_config.get("x_key", None),
                x_vals=plot_config.get("x_vals", None),
                silent_fail=plot_config.get("silent_fail", False),
            )

            variation_map = list(parameterized_grid.keys())
            variation_map.sort()

            fig, axes, plt_lines = plot_grid.plot_grid(
                plot_fn=plot_config.get("plot_fn", plot_cell.make_convergence_plot),
                results=parameterized_grid[variation_map[0]],
                figure_labels=plot_config["figure_labels"],
                line_kwargs=plot_config["line_kwargs"],
                limits=plot_config["limits"],
                log_scale=plot_config["log_scale"],
                base_dir=os.path.join(
                    dest, f"{plot_name}", f"{plot_config['name']}.pdf"
                ),
                settings=plot_config["settings"],
            )

            def update(frame):
                return plot_grid.update_grid(
                    fig,
                    axes,
                    plt_lines,
                    plot_config["update_fn"],
                    parameterized_grid[variation_map[frame]],
                )

            ani = animation.FuncAnimation(
                fig=fig,
                func=update,
                frames=len(variation_map),
                interval=INTERVAL_LENGTH,
            )
            plt.show()

    logger.info("Plotting done.")
