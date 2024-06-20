"""Plotting methods for model evaluation."""

import numpy
import matplotlib.colors
import matplotlib.patches
from matplotlib import pyplot

CSI_LEVELS = numpy.linspace(0, 1, num=11, dtype=float)
PEIRCE_SCORE_LEVELS = numpy.linspace(0, 1, num=11, dtype=float)

ROC_CURVE_COLOUR = numpy.array([228, 26, 28], dtype=float) / 255
PERF_DIAGRAM_COLOUR = numpy.array([228, 26, 28], dtype=float) / 255

FREQ_BIAS_COLOUR = numpy.full(3, 152. / 255)
FREQ_BIAS_WIDTH = 2.
FREQ_BIAS_STRING_FORMAT = '%.2f'
FREQ_BIAS_PADDING = 10
FREQ_BIAS_LEVELS = numpy.array([0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5])

RELIABILITY_LINE_COLOUR = numpy.array([228, 26, 28], dtype=float) / 255
DEFAULT_LINE_WIDTH = 3.

REFERENCE_LINE_COLOUR = numpy.full(3, 152. / 255)
REFERENCE_LINE_WIDTH = 2.

CLIMO_LINE_COLOUR = numpy.full(3, 152. / 255)
CLIMO_LINE_WIDTH = 2.

ZERO_SKILL_LINE_COLOUR = numpy.array([31, 120, 180], dtype=float) / 255
ZERO_SKILL_LINE_WIDTH = 2.
POSITIVE_SKILL_AREA_OPACITY = 0.2

HISTOGRAM_FACE_COLOUR = numpy.array([228, 26, 28], dtype=float) / 255
HISTOGRAM_EDGE_COLOUR = numpy.full(3, 0.)
HISTOGRAM_EDGE_WIDTH = 2.
HISTOGRAM_FONT_SIZE = 16

FONT_SIZE = 20
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)


def _plot_reliability_curve(
        axes_object, mean_predictions, mean_observations, min_value_to_plot,
        max_value_to_plot, line_colour=RELIABILITY_LINE_COLOUR,
        line_style='solid', line_width=DEFAULT_LINE_WIDTH):
    """Plots reliability curve.

    B = number of bins

    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param mean_predictions: length-B numpy array of mean predicted values.
    :param mean_observations: length-B numpy array of mean observed values.
    :param min_value_to_plot: See doc for `plot_attributes_diagram`.
    :param max_value_to_plot: Same.
    :param line_colour: Line colour (in any format accepted by matplotlib).
    :param line_style: Line style (in any format accepted by matplotlib).
    :param line_width: Line width (in any format accepted by matplotlib).
    :return: main_line_handle: Handle for main line (reliability curve).
    """

    perfect_x_coords = numpy.array([min_value_to_plot, max_value_to_plot])
    perfect_y_coords = numpy.array([min_value_to_plot, max_value_to_plot])

    axes_object.plot(
        perfect_x_coords, perfect_y_coords, color=REFERENCE_LINE_COLOUR,
        linestyle='dashed', linewidth=REFERENCE_LINE_WIDTH
    )

    nan_flags = numpy.logical_or(
        numpy.isnan(mean_predictions), numpy.isnan(mean_observations)
    )

    if numpy.all(nan_flags):
        main_line_handle = None
    else:
        real_indices = numpy.where(numpy.invert(nan_flags))[0]

        main_line_handle = axes_object.plot(
            mean_predictions[real_indices], mean_observations[real_indices],
            color=line_colour, linestyle=line_style, linewidth=line_width
        )[0]

    axes_object.set_xlabel('Prediction')
    axes_object.set_ylabel('Conditional mean observation')
    axes_object.set_xlim(min_value_to_plot, max_value_to_plot)
    axes_object.set_ylim(min_value_to_plot, max_value_to_plot)

    return main_line_handle


def _get_positive_skill_area(mean_value_in_training, min_value_in_plot,
                             max_value_in_plot):
    """Returns positive-skill area (where BSS > 0) for attributes diagram.

    :param mean_value_in_training: Mean of target variable in training data.
    :param min_value_in_plot: Minimum value in plot (for both x- and y-axes).
    :param max_value_in_plot: Max value in plot (for both x- and y-axes).
    :return: x_coords_left: length-5 numpy array of x-coordinates for left part
        of positive-skill area.
    :return: y_coords_left: Same but for y-coordinates.
    :return: x_coords_right: length-5 numpy array of x-coordinates for right
        part of positive-skill area.
    :return: y_coords_right: Same but for y-coordinates.
    """

    x_coords_left = numpy.array([
        min_value_in_plot, mean_value_in_training, mean_value_in_training,
        min_value_in_plot, min_value_in_plot
    ])
    y_coords_left = numpy.array([
        min_value_in_plot, min_value_in_plot, mean_value_in_training,
        (min_value_in_plot + mean_value_in_training) / 2, min_value_in_plot
    ])

    x_coords_right = numpy.array([
        mean_value_in_training, max_value_in_plot, max_value_in_plot,
        mean_value_in_training, mean_value_in_training
    ])
    y_coords_right = numpy.array([
        mean_value_in_training,
        (max_value_in_plot + mean_value_in_training) / 2,
        max_value_in_plot, max_value_in_plot, mean_value_in_training
    ])

    return x_coords_left, y_coords_left, x_coords_right, y_coords_right


def _get_zero_skill_line(mean_value_in_training, min_value_in_plot,
                         max_value_in_plot):
    """Returns zero-skill line (where BSS = 0) for attributes diagram.

    :param mean_value_in_training: Mean of target variable in training data.
    :param min_value_in_plot: Minimum value in plot (for both x- and y-axes).
    :param max_value_in_plot: Max value in plot (for both x- and y-axes).
    :return: x_coords: length-2 numpy array of x-coordinates.
    :return: y_coords: Same but for y-coordinates.
    """

    x_coords = numpy.array([min_value_in_plot, max_value_in_plot], dtype=float)
    y_coords = 0.5 * (mean_value_in_training + x_coords)

    return x_coords, y_coords


def _vertex_arrays_to_list(vertex_x_coords, vertex_y_coords):
    """Converts vertices of simple polygon from two arrays to one list.

    V = number of vertices

    :param vertex_x_coords: length-V numpy array of x-coordinates.
    :param vertex_y_coords: length-V numpy array of y-coordinates.
    :return: vertex_coords_as_list: length-V list, where each element is an
        (x, y) tuple.
    """

    num_vertices = len(vertex_x_coords)
    vertex_coords_as_list = []

    for i in range(num_vertices):
        vertex_coords_as_list.append((vertex_x_coords[i], vertex_y_coords[i]))

    return vertex_coords_as_list


def _plot_attr_diagram_background(
        axes_object, mean_value_in_training, min_value_in_plot,
        max_value_in_plot):
    """Plots background (reference lines and polygons) of attributes diagram.

    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param mean_value_in_training: Mean of target variable in training data.
    :param min_value_in_plot: Minimum value in plot (for both x- and y-axes).
    :param max_value_in_plot: Max value in plot (for both x- and y-axes).
    """

    x_coords_left, y_coords_left, x_coords_right, y_coords_right = (
        _get_positive_skill_area(
            mean_value_in_training=mean_value_in_training,
            min_value_in_plot=min_value_in_plot,
            max_value_in_plot=max_value_in_plot
        )
    )

    skill_area_colour = matplotlib.colors.to_rgba(
        ZERO_SKILL_LINE_COLOUR, POSITIVE_SKILL_AREA_OPACITY
    )

    left_polygon_coord_matrix = numpy.transpose(numpy.vstack((
        x_coords_left, y_coords_left
    )))
    left_patch_object = matplotlib.patches.Polygon(
        left_polygon_coord_matrix, lw=0,
        ec=skill_area_colour, fc=skill_area_colour
    )
    axes_object.add_patch(left_patch_object)

    right_polygon_coord_matrix = numpy.transpose(numpy.vstack((
        x_coords_right, y_coords_right
    )))
    right_patch_object = matplotlib.patches.Polygon(
        right_polygon_coord_matrix, lw=0,
        ec=skill_area_colour, fc=skill_area_colour
    )
    axes_object.add_patch(right_patch_object)

    no_skill_x_coords, no_skill_y_coords = _get_zero_skill_line(
        mean_value_in_training=mean_value_in_training,
        min_value_in_plot=min_value_in_plot,
        max_value_in_plot=max_value_in_plot
    )

    axes_object.plot(
        no_skill_x_coords, no_skill_y_coords, color=ZERO_SKILL_LINE_COLOUR,
        linestyle='solid', linewidth=ZERO_SKILL_LINE_WIDTH
    )

    climo_x_coords = numpy.full(2, mean_value_in_training)
    climo_y_coords = numpy.array([min_value_in_plot, max_value_in_plot])
    axes_object.plot(
        climo_x_coords, climo_y_coords, color=CLIMO_LINE_COLOUR,
        linestyle='dashed', linewidth=CLIMO_LINE_WIDTH
    )

    axes_object.plot(
        climo_y_coords, climo_x_coords, color=CLIMO_LINE_COLOUR,
        linestyle='dashed', linewidth=CLIMO_LINE_WIDTH
    )


def _plot_inset_histogram(
        figure_object, bin_centers, bin_counts, has_predictions,
        bar_colour=HISTOGRAM_FACE_COLOUR):
    """Plots histogram as inset in attributes diagram.

    B = number of bins

    :param figure_object: Will plot on this figure (instance of
        `matplotlib.figure.Figure`).
    :param bin_centers: length-B numpy array with value at center of each bin.
        These values will be plotted on the x-axis.
    :param bin_counts: length-B numpy array with number of examples in each bin.
        These values will be plotted on the y-axis.
    :param has_predictions: Boolean flag.  If True, histogram will contain
        prediction frequencies.  If False, will contain observation frequencies.
    :param bar_colour: Bar colour (in any format accepted by matplotlib).
    """

    bin_frequencies = bin_counts.astype(float) / numpy.sum(bin_counts)

    if has_predictions:
        inset_axes_object = figure_object.add_axes([0.675, 0.225, 0.2, 0.2])
    else:
        inset_axes_object = figure_object.add_axes([0.2, 0.625, 0.2, 0.2])

    num_bins = len(bin_centers)
    fake_bin_centers = (
        0.5 + numpy.linspace(0, num_bins - 1, num=num_bins, dtype=float)
    )

    real_indices = numpy.where(numpy.invert(numpy.isnan(bin_centers)))[0]

    inset_axes_object.bar(
        fake_bin_centers[real_indices], bin_frequencies[real_indices], 1.,
        color=bar_colour, edgecolor=HISTOGRAM_EDGE_COLOUR,
        linewidth=HISTOGRAM_EDGE_WIDTH
    )
    inset_axes_object.set_ylim(bottom=0.)

    tick_indices = []

    for i in real_indices:
        if numpy.mod(i, 2) == 0:
            tick_indices.append(i)
            continue

        if i - 1 in real_indices or i + 1 in real_indices:
            continue

        tick_indices.append(i)

    x_tick_values = fake_bin_centers[tick_indices]
    x_tick_labels = ['{0:.2g}'.format(b) for b in bin_centers[tick_indices]]
    inset_axes_object.set_xticks(x_tick_values)
    inset_axes_object.set_xticklabels(x_tick_labels)

    for this_tick_object in inset_axes_object.xaxis.get_major_ticks():
        this_tick_object.label.set_fontsize(HISTOGRAM_FONT_SIZE)
        this_tick_object.label.set_rotation('vertical')

    for this_tick_object in inset_axes_object.yaxis.get_major_ticks():
        this_tick_object.label.set_fontsize(HISTOGRAM_FONT_SIZE)

    inset_axes_object.set_title(
        'Prediction frequency' if has_predictions else 'Observation frequency',
        fontsize=HISTOGRAM_FONT_SIZE
    )


def _get_pofd_pod_grid(pofd_spacing=0.01, pod_spacing=0.01):
    """Creates grid in POFD-POD space.

    POFD = probability of false detection
    POD = probability of detection

    M = number of rows (unique POD values) in grid
    N = number of columns (unique POFD values) in grid

    :param pofd_spacing: Spacing between grid cells in adjacent columns.
    :param pod_spacing: Spacing between grid cells in adjacent rows.
    :return: pofd_matrix: M-by-N numpy array of POFD values.
    :return: pod_matrix: M-by-N numpy array of POD values.
    """

    num_pofd_values = int(numpy.ceil(1. / pofd_spacing))
    num_pod_values = int(numpy.ceil(1. / pod_spacing))

    unique_pofd_values = numpy.linspace(
        0, 1, num=num_pofd_values + 1, dtype=float
    )
    unique_pofd_values = unique_pofd_values[:-1] + pofd_spacing / 2

    unique_pod_values = numpy.linspace(
        0, 1, num=num_pod_values + 1, dtype=float
    )
    unique_pod_values = unique_pod_values[:-1] + pod_spacing / 2

    return numpy.meshgrid(unique_pofd_values, unique_pod_values[::-1])


def _get_peirce_colour_scheme():
    """Returns colour scheme for Peirce score.

    :return: colour_map_object: Colour scheme (instance of
        `matplotlib.colors.ListedColormap`).
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`,
        defining the scale of the colour map.
    """

    this_colour_map_object = pyplot.get_cmap('Blues')
    this_colour_norm_object = matplotlib.colors.BoundaryNorm(
        PEIRCE_SCORE_LEVELS, this_colour_map_object.N
    )

    rgba_matrix = this_colour_map_object(this_colour_norm_object(
        PEIRCE_SCORE_LEVELS
    ))
    colour_list = [
        rgba_matrix[i, ..., :-1] for i in range(rgba_matrix.shape[0])
    ]

    colour_map_object = matplotlib.colors.ListedColormap(colour_list)
    colour_map_object.set_under(numpy.full(3, 1.))
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        PEIRCE_SCORE_LEVELS, colour_map_object.N
    )

    return colour_map_object, colour_norm_object


def _get_sr_pod_grid(success_ratio_spacing=0.01, pod_spacing=0.01):
    """Creates grid in SR-POD space

    SR = success ratio
    POD = probability of detection

    M = number of rows (unique POD values) in grid
    N = number of columns (unique success ratios) in grid

    :param success_ratio_spacing: Spacing between adjacent success ratios
        (x-values) in grid.
    :param pod_spacing: Spacing between adjacent POD values (y-values) in grid.
    :return: success_ratio_matrix: M-by-N numpy array of success ratios.
        Success ratio increases while traveling right along a row.
    :return: pod_matrix: M-by-N numpy array of POD values.  POD increases while
        traveling up a column.
    """

    num_success_ratios = int(numpy.ceil(1. / success_ratio_spacing))
    num_pod_values = int(numpy.ceil(1. / pod_spacing))

    unique_success_ratios = numpy.linspace(
        0, 1, num=num_success_ratios + 1, dtype=float
    )
    unique_success_ratios = (
        unique_success_ratios[:-1] + success_ratio_spacing / 2
    )

    unique_pod_values = numpy.linspace(
        0, 1, num=num_pod_values + 1, dtype=float
    )
    unique_pod_values = unique_pod_values[:-1] + pod_spacing / 2

    return numpy.meshgrid(unique_success_ratios, unique_pod_values[::-1])


def _csi_from_sr_and_pod(success_ratio_array, pod_array):
    """Computes CSI (critical success index) from success ratio and POD.

    POD = probability of detection

    :param success_ratio_array: numpy array (any shape) of success ratios.
    :param pod_array: numpy array (same shape) of POD values.
    :return: csi_array: numpy array (same shape) of CSI values.
    """

    return (success_ratio_array ** -1 + pod_array ** -1 - 1.) ** -1


def _bias_from_sr_and_pod(success_ratio_array, pod_array):
    """Computes frequency bias from success ratio and POD.

    POD = probability of detection

    :param success_ratio_array: numpy array (any shape) of success ratios.
    :param pod_array: numpy array (same shape) of POD values.
    :return: frequency_bias_array: numpy array (same shape) of frequency biases.
    """

    return pod_array / success_ratio_array


def _get_csi_colour_scheme():
    """Returns colour scheme for CSI (critical success index).

    :return: colour_map_object: Colour scheme (instance of
        `matplotlib.colors.ListedColormap`).
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`,
        defining the scale of the colour map.
    """

    this_colour_map_object = pyplot.get_cmap('Blues')
    this_colour_norm_object = matplotlib.colors.BoundaryNorm(
        CSI_LEVELS, this_colour_map_object.N
    )

    rgba_matrix = this_colour_map_object(this_colour_norm_object(
        CSI_LEVELS
    ))
    colour_list = [
        rgba_matrix[i, ..., :-1] for i in range(rgba_matrix.shape[0])
    ]

    colour_map_object = matplotlib.colors.ListedColormap(colour_list)
    colour_map_object.set_under(numpy.full(3, 1.))
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        CSI_LEVELS, colour_map_object.N
    )

    return colour_map_object, colour_norm_object


def _add_colour_bar(
        axes_object, colour_map_object, values_to_colour, min_colour_value=None,
        max_colour_value=None, colour_norm_object=None,
        orientation_string='vertical', extend_min=True, extend_max=True):
    """Adds colour bar to existing axes.

    :param axes_object: Existing axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param colour_map_object: Colour scheme (instance of
        `matplotlib.pyplot.cm`).
    :param values_to_colour: numpy array of values to colour.
    :param min_colour_value: Minimum value in colour scheme.
    :param max_colour_value: Max value in colour scheme.
    :param colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`,
        defining the scale of the colour map.  If `colour_norm_object is None`,
        will assume that scale is linear.
    :param orientation_string: Orientation of colour bar ("vertical" or
        "horizontal").
    :param extend_min: Boolean flag.  If True, the bottom of the colour bar will
        have an arrow.  If False, it will be a flat line, suggesting that lower
        values are not possible.
    :param extend_max: Same but for top of colour bar.
    :return: colour_bar_object: Colour bar (instance of
        `matplotlib.pyplot.colorbar`) created by this method.
    """

    # TODO(thunderhoser): Stop duplicating this between here and utils.py.
    # TODO(thunderhoser): And replace this with better method.

    if colour_norm_object is None:
        colour_norm_object = matplotlib.colors.Normalize(
            vmin=min_colour_value, vmax=max_colour_value, clip=False
        )

    scalar_mappable_object = pyplot.cm.ScalarMappable(
        cmap=colour_map_object, norm=colour_norm_object
    )
    scalar_mappable_object.set_array(values_to_colour)

    if extend_min and extend_max:
        extend_string = 'both'
    elif extend_min:
        extend_string = 'min'
    elif extend_max:
        extend_string = 'max'
    else:
        extend_string = 'neither'

    if orientation_string == 'horizontal':
        padding = 0.075
    else:
        padding = 0.05

    colour_bar_object = pyplot.colorbar(
        ax=axes_object, mappable=scalar_mappable_object,
        orientation=orientation_string, pad=padding, extend=extend_string,
        shrink=0.8
    )

    colour_bar_object.ax.tick_params(labelsize=FONT_SIZE)
    return colour_bar_object


def plot_attributes_diagram(
        figure_object, axes_object, mean_predictions, mean_observations,
        example_counts, mean_value_in_training, min_value_to_plot,
        max_value_to_plot, line_colour=RELIABILITY_LINE_COLOUR,
        line_style='solid', line_width=DEFAULT_LINE_WIDTH,
        inv_mean_observations=None, inv_example_counts=None):
    """Plots attributes diagram.

    If `inv_mean_observations is None` and `inv_example_counts is None`, this
    method will plot only the histogram of predicted values, not the histogram
    of observed values.

    B = number of bins

    :param figure_object: Will plot on this figure (instance of
        `matplotlib.figure.Figure`).
    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param mean_predictions: length-B numpy array of mean predicted values.
    :param mean_observations: length-B numpy array of mean observed values.
    :param example_counts: length-B numpy array with number of examples in each
        bin.
    :param mean_value_in_training: Mean of target variable in training data.
    :param min_value_to_plot: Minimum value in plot (for both x- and y-axes).
    :param max_value_to_plot: Max value in plot (for both x- and y-axes).
        If None, will be determined automatically.
    :param line_colour: See doc for `_plot_reliability_curve`.
    :param line_width: Same.
    :param line_style: Same.
    :param inv_mean_observations: length-B numpy array of mean observed values
        for inverted reliability curve.
    :param inv_example_counts: length-B numpy array of example counts for
        inverted reliability curve.
    :return: main_line_handle: See doc for `_plot_reliability_curve`.
    """

    if max_value_to_plot == min_value_to_plot:
        max_value_to_plot = min_value_to_plot + 1.

    plot_obs_histogram = not(
        inv_mean_observations is None and inv_example_counts is None
    )

    _plot_attr_diagram_background(
        axes_object=axes_object, mean_value_in_training=mean_value_in_training,
        min_value_in_plot=min_value_to_plot, max_value_in_plot=max_value_to_plot
    )

    _plot_inset_histogram(
        figure_object=figure_object, bin_centers=mean_predictions,
        bin_counts=example_counts, has_predictions=True, bar_colour=line_colour
    )

    if plot_obs_histogram:
        _plot_inset_histogram(
            figure_object=figure_object, bin_centers=inv_mean_observations,
            bin_counts=inv_example_counts, has_predictions=False,
            bar_colour=line_colour
        )

    return _plot_reliability_curve(
        axes_object=axes_object, mean_predictions=mean_predictions,
        mean_observations=mean_observations,
        min_value_to_plot=min_value_to_plot,
        max_value_to_plot=max_value_to_plot,
        line_colour=line_colour, line_style=line_style, line_width=line_width
    )


def plot_roc_curve(axes_object, pod_by_threshold, pofd_by_threshold,
                   line_colour=ROC_CURVE_COLOUR, plot_background=True):
    """Plots ROC (receiver operating characteristic) curve.

    T = number of probability thresholds

    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param pod_by_threshold: length-T numpy array of POD (probability of
        detection) values.
    :param pofd_by_threshold: length-T numpy array of POFD (probability of false
        detection) values.
    :param line_colour: Line colour.
    :param plot_background: Boolean flag.  If True, will plot background
        (reference line and Peirce-score contours).
    :return: line_handle: Line handle for ROC curve.
    """

    if plot_background:
        pofd_matrix, pod_matrix = _get_pofd_pod_grid()
        peirce_score_matrix = pod_matrix - pofd_matrix

        this_colour_map_object, this_colour_norm_object = (
            _get_peirce_colour_scheme()
        )

        pyplot.contourf(
            pofd_matrix, pod_matrix, peirce_score_matrix, PEIRCE_SCORE_LEVELS,
            cmap=this_colour_map_object, norm=this_colour_norm_object, vmin=0.,
            vmax=1., axes=axes_object
        )

        colour_bar_object = _add_colour_bar(
            axes_object=axes_object, values_to_colour=peirce_score_matrix,
            colour_map_object=this_colour_map_object,
            colour_norm_object=this_colour_norm_object,
            orientation_string='vertical', extend_min=False, extend_max=False
        )

        colour_bar_object.set_label('Peirce score (POD minus POFD)')

        random_x_coords = numpy.array([0, 1], dtype=float)
        random_y_coords = numpy.array([0, 1], dtype=float)
        axes_object.plot(
            random_x_coords, random_y_coords, color=REFERENCE_LINE_COLOUR,
            linestyle='dashed', linewidth=REFERENCE_LINE_WIDTH
        )

    nan_flags = numpy.logical_or(
        numpy.isnan(pofd_by_threshold), numpy.isnan(pod_by_threshold)
    )

    if numpy.all(nan_flags):
        line_handle = None
    else:
        real_indices = numpy.where(numpy.invert(nan_flags))[0]

        line_handle = axes_object.plot(
            pofd_by_threshold[real_indices], pod_by_threshold[real_indices],
            color=line_colour, linestyle='solid', linewidth=DEFAULT_LINE_WIDTH
        )[0]

    axes_object.set_xlabel('POFD (probability of false detection)')
    axes_object.set_ylabel('POD (probability of detection)')
    axes_object.set_xlim(0., 1.)
    axes_object.set_ylim(0., 1.)

    return line_handle


def plot_performance_diagram(
        axes_object, pod_by_threshold, success_ratio_by_threshold,
        line_colour=PERF_DIAGRAM_COLOUR, plot_background=True):
    """Plots performance diagram.

    T = number of probability thresholds

    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param pod_by_threshold: length-T numpy array of POD (probability of
        detection) values.
    :param success_ratio_by_threshold: length-T numpy array of success ratios.
    :param line_colour: Line colour.
    :param plot_background: Boolean flag.  If True, will plot background
        (frequency-bias and CSI contours).
    :return: line_handle: Line handle for ROC curve.
    """

    if plot_background:
        success_ratio_matrix, pod_matrix = _get_sr_pod_grid()
        csi_matrix = _csi_from_sr_and_pod(
            success_ratio_array=success_ratio_matrix, pod_array=pod_matrix
        )
        frequency_bias_matrix = _bias_from_sr_and_pod(
            success_ratio_array=success_ratio_matrix, pod_array=pod_matrix
        )

        this_colour_map_object, this_colour_norm_object = (
            _get_csi_colour_scheme()
        )
        pyplot.contourf(
            success_ratio_matrix, pod_matrix, csi_matrix, CSI_LEVELS,
            cmap=this_colour_map_object, norm=this_colour_norm_object, vmin=0.,
            vmax=1., axes=axes_object
        )

        colour_bar_object = _add_colour_bar(
            axes_object=axes_object, values_to_colour=csi_matrix,
            colour_map_object=this_colour_map_object,
            colour_norm_object=this_colour_norm_object,
            orientation_string='vertical', extend_min=False, extend_max=False
        )
        colour_bar_object.set_label('CSI (critical success index)')

        bias_colour_tuple = tuple(FREQ_BIAS_COLOUR.tolist())
        bias_colours_2d_tuple = ()
        for _ in range(len(FREQ_BIAS_LEVELS)):
            bias_colours_2d_tuple += (bias_colour_tuple,)

        bias_contour_object = pyplot.contour(
            success_ratio_matrix, pod_matrix, frequency_bias_matrix,
            FREQ_BIAS_LEVELS, colors=bias_colours_2d_tuple,
            linewidths=FREQ_BIAS_WIDTH, linestyles='dashed', axes=axes_object
        )
        axes_object.clabel(
            bias_contour_object, inline=True, inline_spacing=FREQ_BIAS_PADDING,
            fmt=FREQ_BIAS_STRING_FORMAT, fontsize=FONT_SIZE
        )

    nan_flags = numpy.logical_or(
        numpy.isnan(success_ratio_by_threshold), numpy.isnan(pod_by_threshold)
    )

    if numpy.all(nan_flags):
        line_handle = None
    else:
        real_indices = numpy.where(numpy.invert(nan_flags))[0]

        line_handle = axes_object.plot(
            success_ratio_by_threshold[real_indices],
            pod_by_threshold[real_indices], color=line_colour,
            linestyle='solid', linewidth=DEFAULT_LINE_WIDTH
        )[0]

    axes_object.set_xlabel('Success ratio (1 - FAR)')
    axes_object.set_ylabel('POD (probability of detection)')
    axes_object.set_xlim(0., 1.)
    axes_object.set_ylim(0., 1.)

    return line_handle
