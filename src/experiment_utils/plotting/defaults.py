"""
Plotting defaults
"""

DEFAULT_SETTINGS = {
    # SIZES
    # figure
    "fig_width": 6,  # width of each sub-figure
    "fig_height": 6,  # height of each sub-figure
    # fonts:
    "titles_fs": 32,  # font-size for plot title.
    "subtitle_fs": 30,  # font-size for sub-titles (e.g. column titles)f
    "axis_labels_fs": 24,  # font-size for axis labels
    "legend_fs": 22,  # font-size for text in the legend.
    "tick_fs": 14,  # font-size for ticks.
    "offest_text_fs": 12,  # font-size for y-offset. (printed at top of y-axis when using scientific notation)
    "row_title_pad": 5,  # padding for row-titles. May need to be tweaked for big row titles.
    # bar charts:
    "bar_width": 0.75,  # width for bars in a bar chart.
    "bar_labels": False,  # whether or not to put labels beneath each bar.
    "capsize": 10,  # size for error-bar caps used in the bar-chart.
    # plots:
    "error_alpha": 0.2,  # alpha for shaded error bars.
    "marker_frequency": 50,  # f requency for placing marks in plot.
    "marker_size": 14,  # size for markers in plot.
    "line_width": 4,  # width for lines in the plot.
    "line_alpha": 0.9,  # alpha for the lines themselves
    "bottom_margin": 0.22,
    "wspace": 0.3,
    "vspace": 0.15,
    # OPTIONS
    # titles and labels
    "col_titles": True,  # Whether or not to title the columns: True OR False
    "row_titles": True,  # Whether or not to title the rows: True OR False
    "y_labels": "left_col",  # "every_col",   # Where to put y-axis labels: "left_col" OR "every_col" OR False/None
    "x_labels": "bottom_row",  # Where to put x-axis labels: "bottom_col" OR "every_row" OR False/None
    "legend_cols": 4,  # number of columns for the legend
    "legend_lw": 4.0,
    "show_legend": True,
    "ticklabel_format": "plain",
}

# default colors for lines.
line_colors = [
    "#000000",
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#8c564b",
    "#17becf",
    "#556B2F",
    "#FFFF00",
    "#191970",
]

# default line-style is solid.
line_styles = [
    "solid",
    "dashed",
    "dashdot",
    "dotted",
]
# default marker styles.
marker_styles = ["o", "s", "v", "X", "D", "^", "D", "p", "o", "x", "s"]


def get_default_line_kwargs(lines):
    line_kwargs = {}
    for i, line in enumerate(lines):
        line_kwargs[line] = {
            "c": line_colors[i % len(line_colors)],
            "label": line,
            "linewidth": 3,
            "marker": marker_styles[i % len(marker_styles)],
            "markevery": 0.1,
            "markersize": 8,
            "linestyle": line_styles[i // len(line_colors)],
        }

    return line_kwargs
