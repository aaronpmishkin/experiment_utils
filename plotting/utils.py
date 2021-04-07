"""
Utilities for plotting.
"""


def try_cell_row_col(value_dict, row, col, default_value=None):
    """ Helper for finding the dictionary value associated with a cell in the plot.
    The dictionary is first index by cell using `(row, col)`, then by row using `row`,
    and finally by column using `col`. `default_value` is returned if nothing is found.
    :param value_dict: dictionary to index into. e.g y-axis labels for each cell.
    :param row: key for the plot row.
    :param col: key for the plot column.
    :param default_value: value to return if nothing is found. Optional. Defaults to `None`.
    :returns: value_dict[(row, col)], value_dict[row], value_dict[col] or `default_value`.
    """
    return value_dict.get((row, col),
                          value_dict.get(row,
                                         value_dict.get(col,
                                                        default_value)))


# TODO: replace plt.* with axis.*?

def make_convergence_plot(ax, results, lines, uncertainty_type, line_kwargs, settings, projected_results=None, log_freq=None, marker_freq=None):
    """ Generate convergence plot with optional error bars for methods indexed by `lines`.
    Arguments:
        ax: axis object.
        results: dict of dicts. The first level is indexed by the keys in `lines`.
            The second level contains summary statistics indexed by `mean`, `std`, `median`,
            `1_quartile` and `3_quartile`.
        uncertainty_type: `quartiles`, `std`, or `None`.
        line_kwargs: dict of key-work arguments for each bar.
        settings: configuration object for plot. See DEFAULT_SETTINGS above.
    """
    # Iterate over the experiments - one line per exp
    if log_freq is None:
        log_freq = settings['log_freq']

    for i, line in enumerate(lines):
        y = results[line]["median"]
        x = np.arange(len(y)*log_freq, step=log_freq)
        if uncertainty_type == "std":
            std = results[line]["std"]
            plt.fill_between(x, y-std, y+std, alpha=settings["error_alpha"], color=line_kwargs[line]['c'])

            if projected_results is not None and 'proj_True' not in line:
                left, right = plt.xlim()
                offset = (right - left) / 25

                y_proj = projected_results[line]["mean"]

                plt.plot([right - offset], y_proj, marker='*', alpha=0.5, markersize=settings['proj_marker_size'], color=line_kwargs[line]['c'])
                plt.plot(x, np.repeat(y_proj, len(x)), linewidth=4, marker='', alpha=0.6, linestyle=':', color=line_kwargs[line]['c'])

        elif uncertainty_type == "quartiles":
            y = results[line]["median"]
            first_quartile = results[line]["1_quartile"]
            third_quartile = results[line]["3_quartile"]
            plt.fill_between(x, first_quartile, third_quartile, alpha=settings["error_alpha"], color=line_kwargs[line]['c'])

        kwargs = line_kwargs[line].copy()
        if marker_freq is not None:
            kwargs["markevery"] = marker_freq

        plt.plot(x, y, alpha=settings['line_alpha'], **kwargs)
