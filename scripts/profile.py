"""
Plot an experiment.
"""

import os
from typing import List, Dict
from functools import reduce
from collections import defaultdict

import numpy as np

from experiment_utils import configs, utils, files, command_line
from experiment_utils.plotting import plot_grid, plot_cell, defaults
from experiment_utils.performance_profile import (
    compute_success_ratios,
    compute_xy_values_tol,
)


# merge experiment dictionaries.
from exp_configs import EXPERIMENTS
from plot_configs import PLOTS

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
    config_list = configs.expand_config_list(config_list)
    results_dir = [os.path.join(results_dir_base, eid) for eid in exp_ids]

    logger.info(f"\n\n====== Making {plot_names} from {exp_ids} results ======\n")

    for plot_name in plot_names:
        dest = os.path.join(figures_dir, "_".join(exp_ids))
        plot_config_list = PLOTS[plot_name]

        for i, plot_config in enumerate(plot_config_list):
            logger.info(f"\n\n Creating {i+1}/{len(plot_config_list)} \n")
            filtered_config_list = configs.filter_dict_list(
                config_list,
                keep=["keep"],
                remove=["remove"],
                filter_fn=plot_config["filter_fn"],
            )

            config_grid = configs.make_performance_profile_grid(
                filtered_config_list,
                row_key=plot_config["row_key"],
                col_key=plot_config["col_key"],
                problem_key=plot_config["problem_key"],
                line_key=plot_config["line_key"],
            )

            def processing_fn(exp, key):
                for fn in plot_config["processing_fns"]:
                    exp = fn(exp, key)
                return exp

            target_metric = plot_config["target_metric"]
            maximize_target = plot_config.get("maximize_target", False)
            replace_val = -np.inf if maximize_target else np.inf
            x_key = plot_config.get("x_key", None)

            metric_grid = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
            best_target = defaultdict(lambda: defaultdict(dict))
            for row in config_grid.keys():
                for col in config_grid[row].keys():
                    for problem in config_grid[row][col].keys():
                        best_target_vals = []
                        for line in config_grid[row][col][problem].keys():
                            exp_config = config_grid[row][col][problem][line]
                            try:
                                metrics = files.load_experiment(
                                    exp_dict=exp_config,
                                    results_dir=results_dir,
                                    load_metrics=True,
                                    load_model=False,
                                )["metrics"]
                            except Exception as e:
                                if plot_config["silent_fail"]:
                                    continue

                                raise e
                            vals = metrics[target_metric]
                            vals = processing_fn(
                                vals,
                                (row, col, line, target_metric),
                            )

                            if x_key is not None:
                                x = metrics[x_key]
                                x = processing_fn(
                                    x,
                                    (row, col, line, x_key),
                                )
                            else:
                                x = np.arange(0, len(vals))

                            metric_grid[row][col][problem][line] = (x, vals)

                            if maximize_target:
                                best_target_vals.append(np.max(vals))
                            else:
                                best_target_vals.append(np.min(vals))

                        best_target_vals = np.nan_to_num(
                            best_target_vals,
                            nan=replace_val,
                        )
                        if maximize_target:
                            best_target[row][col][problem] = np.max(best_target_vals)
                        else:
                            best_target[row][col][problem] = np.min(best_target_vals)

            # compute success ratios
            success_ratio_grid = defaultdict(lambda: defaultdict(dict))
            for row in metric_grid.keys():
                for col in metric_grid.keys():
                    success_ratio_grid[row][col] = compute_success_ratios(
                        metric_grid[row][col],
                        compute_xy_values_tol,
                        best_target,
                        plot_config["tol"],
                        maximize_target,
                    )

            # call plot_grid on success ratios
            plot_grid.plot_grid(
                plot_fn=plot_config.get("plot_fn", plot_cell.make_convergence_plot),
                results=success_ratio_grid,
                figure_labels=plot_config["figure_labels"],
                line_kwargs=plot_config["line_kwargs"],
                limits=plot_config["limits"],
                log_scale=plot_config["log_scale"],
                base_dir=os.path.join(
                    dest, f"{plot_name}", f"{plot_config['name']}.pdf"
                ),
                settings=plot_config["settings"],
            )

    logger.info("Plotting done.")
