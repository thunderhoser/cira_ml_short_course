"""Plots results of permutation importance test."""

import numpy
from matplotlib import pyplot
from cira_ml_short_course.utils import utils

DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_FACE_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
NO_PERMUTATION_COLOUR = numpy.full(3, 1.)

BAR_EDGE_WIDTH = 2
BAR_EDGE_COLOUR = numpy.full(3, 0.)

REFERENCE_LINE_WIDTH = 4
REFERENCE_LINE_COLOUR = numpy.full(3, 152. / 255)

ERROR_BAR_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
ERROR_BAR_CAP_SIZE = 8
ERROR_BAR_DICT = {'alpha': 1., 'linewidth': 4, 'capthick': 4}

BAR_TEXT_COLOUR = numpy.full(3, 0.)
BAR_FONT_SIZE = 18
DEFAULT_FONT_SIZE = 30
FIGURE_WIDTH_INCHES = 10
FIGURE_HEIGHT_INCHES = 10

pyplot.rc('font', size=DEFAULT_FONT_SIZE)
pyplot.rc('axes', titlesize=DEFAULT_FONT_SIZE)
pyplot.rc('axes', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('xtick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('ytick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('legend', fontsize=DEFAULT_FONT_SIZE)
pyplot.rc('figure', titlesize=DEFAULT_FONT_SIZE)


def _label_bars(axes_object, y_tick_coords, y_tick_strings):
    """Labels bars in graph.

    N = number of bars

    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param y_tick_coords: length-N numpy array with y-coordinates of bars.
    :param y_tick_strings: length-N list of labels.
    """

    for j in range(len(y_tick_coords)):
        axes_object.text(
            0., y_tick_coords[j], '      ' + y_tick_strings[j],
            color=BAR_TEXT_COLOUR, fontsize=BAR_FONT_SIZE,
            horizontalalignment='left', verticalalignment='center'
        )


def _get_error_matrix(cost_matrix, confidence_level):
    """Creates error matrix (used to plot error bars).

    S = number of steps in permutation test
    B = number of bootstrap replicates

    :param cost_matrix: S-by-B numpy array of costs.
    :param confidence_level: Confidence level (in range 0...1).
    :return: error_matrix: 2-by-S numpy array, where the first row contains
        negative errors and second row contains positive errors.
    """

    mean_costs = numpy.mean(cost_matrix, axis=-1)
    min_costs = numpy.percentile(
        cost_matrix, 50 * (1. - confidence_level), axis=-1
    )
    max_costs = numpy.percentile(
        cost_matrix, 50 * (1. + confidence_level), axis=-1
    )

    negative_errors = mean_costs - min_costs
    positive_errors = max_costs - mean_costs

    negative_errors = numpy.reshape(negative_errors, (1, negative_errors.size))
    positive_errors = numpy.reshape(positive_errors, (1, positive_errors.size))

    return numpy.vstack((negative_errors, positive_errors))


def _plot_bars(
        cost_matrix, clean_cost_array, predictor_names, backwards_flag,
        multipass_flag, confidence_level, axes_object, bar_face_colour):
    """Plots bar graph for either single-pass or multi-pass test.

    P = number of predictors permuted or depermuted
    B = number of bootstrap replicates

    :param cost_matrix: (P + 1)-by-B numpy array of costs.  The first row
        contains costs at the beginning of the test -- before (de)permuting any
        variables -- and the [i]th row contains costs after (de)permuting the
        variable represented by predictor_names[i - 1].
    :param clean_cost_array: length-B numpy array of costs with clean
        (depermuted) predictors.
    :param predictor_names: length-P list of predictor names (used to label
        bars).
    :param backwards_flag: Boolean flag.  If True, will plot backwards version
        of permutation, where each step involves *de*permuting a variable.  If
        False, will plot forward version, where each step involves permuting a
        variable.
    :param multipass_flag: Boolean flag.  If True, plotting multi-pass version
        of test.  If False, plotting single-pass version.
    :param confidence_level: Confidence level for error bars (in range 0...1).
    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).  If None, will create new
        axes.
    :param bar_face_colour: Interior colour (in any format accepted by
        matplotlib), used for each bar in the graph.
    """

    mean_clean_cost = numpy.mean(clean_cost_array)
    is_cost_auc = numpy.any(cost_matrix < 0)

    if is_cost_auc:
        cost_matrix *= -1
        mean_clean_cost *= -1
        x_label_string = 'Area under ROC curve'
    else:
        x_label_string = 'Cross-entropy'

    if backwards_flag:
        y_tick_strings = ['All permuted'] + predictor_names
    else:
        y_tick_strings = ['None permuted'] + predictor_names

    y_tick_coords = numpy.linspace(
        0, len(y_tick_strings) - 1, num=len(y_tick_strings), dtype=float
    )

    if multipass_flag:
        y_tick_coords = y_tick_coords[::-1]

    if axes_object is None:
        _, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

    mean_costs = numpy.mean(cost_matrix, axis=-1)
    num_bootstrap_reps = cost_matrix.shape[1]

    if num_bootstrap_reps > 1:
        error_matrix = _get_error_matrix(
            cost_matrix=cost_matrix, confidence_level=confidence_level
        )

        x_min = numpy.min(mean_costs - error_matrix[0, :])
        x_max = numpy.max(mean_costs + error_matrix[1, :])

        axes_object.barh(
            y_tick_coords, mean_costs, color=bar_face_colour,
            edgecolor=BAR_EDGE_COLOUR, linewidth=BAR_EDGE_WIDTH,
            xerr=error_matrix, ecolor=ERROR_BAR_COLOUR,
            capsize=ERROR_BAR_CAP_SIZE, error_kw=ERROR_BAR_DICT
        )
    else:
        x_min = numpy.min(mean_costs)
        x_max = numpy.max(mean_costs)

        axes_object.barh(
            y_tick_coords, mean_costs, color=bar_face_colour,
            edgecolor=BAR_EDGE_COLOUR, linewidth=BAR_EDGE_WIDTH
        )

    reference_x_coords = numpy.full(2, mean_clean_cost)
    reference_y_tick_coords = numpy.array([
        numpy.min(y_tick_coords) - 0.75, numpy.max(y_tick_coords) + 0.75
    ])

    axes_object.plot(
        reference_x_coords, reference_y_tick_coords,
        color=REFERENCE_LINE_COLOUR, linestyle='--',
        linewidth=REFERENCE_LINE_WIDTH
    )

    axes_object.set_yticks([], [])
    axes_object.set_xlabel(x_label_string)

    if backwards_flag:
        axes_object.set_ylabel('Variable cleaned')
    else:
        axes_object.set_ylabel('Variable permuted')

    axes_object.set_ylim(
        numpy.min(y_tick_coords) - 0.75, numpy.max(y_tick_coords) + 0.75
    )

    x_max *= 1.01
    if x_min <= 0:
        x_min *= 1.01
    else:
        x_min = 0.

    axes_object.set_xlim(x_min, x_max)

    _label_bars(
        axes_object=axes_object, y_tick_coords=y_tick_coords,
        y_tick_strings=y_tick_strings
    )

    axes_object.set_ylim(
        numpy.min(y_tick_coords) - 0.75, numpy.max(y_tick_coords) + 0.75
    )


def plot_single_pass_test(
        result_dict, bar_face_colour=DEFAULT_FACE_COLOUR,
        confidence_level=DEFAULT_CONFIDENCE_LEVEL,
        axes_object=None, num_predictors_to_plot=None):
    """Plots results of single-pass permutation test.

    :param result_dict: Dictionary created by `utils.run_forward_test` or
        `utils.run_backwards_test`.
    :param bar_face_colour: See doc for `_plot_bars`.
    :param confidence_level: Same.
    :param axes_object: Same.
    :param num_predictors_to_plot: Number of predictors to plot.  Will plot only
        the K most important, where K = `num_predictors_to_plot`.  If None, will
        plot all predictors.
    """

    # Check input args.
    predictor_names = result_dict[utils.STEP1_PREDICTORS_KEY]
    if num_predictors_to_plot is None:
        num_predictors_to_plot = len(predictor_names)

    num_predictors_to_plot = max([num_predictors_to_plot, 2])
    num_predictors_to_plot = min([
        num_predictors_to_plot, len(predictor_names)
    ])

    # Set up plotting args.
    backwards_flag = result_dict[utils.BACKWARDS_FLAG_KEY]
    perturbed_cost_matrix = result_dict[utils.STEP1_COSTS_KEY]
    mean_perturbed_costs = numpy.mean(perturbed_cost_matrix, axis=-1)

    if backwards_flag:
        sort_indices = numpy.argsort(
            mean_perturbed_costs
        )[:num_predictors_to_plot][::-1]
    else:
        sort_indices = numpy.argsort(
            mean_perturbed_costs
        )[-num_predictors_to_plot:]

    perturbed_cost_matrix = perturbed_cost_matrix[sort_indices, :]
    predictor_names = [predictor_names[k] for k in sort_indices]

    original_cost_array = result_dict[utils.ORIGINAL_COST_KEY]
    original_cost_matrix = numpy.reshape(
        original_cost_array, (1, original_cost_array.size)
    )
    cost_matrix = numpy.concatenate(
        (original_cost_matrix, perturbed_cost_matrix), axis=0
    )

    # Do plotting.
    if backwards_flag:
        clean_cost_array = result_dict[utils.BEST_COSTS_KEY][-1, :]
    else:
        clean_cost_array = original_cost_array

    _plot_bars(
        cost_matrix=cost_matrix, clean_cost_array=clean_cost_array,
        predictor_names=predictor_names,
        backwards_flag=backwards_flag, multipass_flag=False,
        confidence_level=confidence_level, axes_object=axes_object,
        bar_face_colour=bar_face_colour
    )


def plot_multipass_test(
        result_dict, bar_face_colour=DEFAULT_FACE_COLOUR,
        confidence_level=DEFAULT_CONFIDENCE_LEVEL,
        axes_object=None, num_predictors_to_plot=None):
    """Plots results of multi-pass permutation test.

    :param result_dict: See doc for `plot_single_pass_test`.
    :param bar_face_colour: Same.
    :param confidence_level: Same.
    :param axes_object: Same.
    :param num_predictors_to_plot: Same.
    """

    # Check input args.
    predictor_names = result_dict[utils.BEST_PREDICTORS_KEY]
    if num_predictors_to_plot is None:
        num_predictors_to_plot = len(predictor_names)

    num_predictors_to_plot = max([num_predictors_to_plot, 2])
    num_predictors_to_plot = min([
        num_predictors_to_plot, len(predictor_names)
    ])

    # Set up plotting args.
    backwards_flag = result_dict[utils.BACKWARDS_FLAG_KEY]
    perturbed_cost_matrix = result_dict[utils.BEST_COSTS_KEY]

    perturbed_cost_matrix = perturbed_cost_matrix[:num_predictors_to_plot, :]
    predictor_names = predictor_names[:num_predictors_to_plot]

    original_cost_array = result_dict[utils.ORIGINAL_COST_KEY]
    original_cost_matrix = numpy.reshape(
        original_cost_array, (1, original_cost_array.size)
    )
    cost_matrix = numpy.concatenate(
        (original_cost_matrix, perturbed_cost_matrix), axis=0
    )

    # Do plotting.
    if backwards_flag:
        clean_cost_array = result_dict[utils.BEST_COSTS_KEY][-1, :]
    else:
        clean_cost_array = original_cost_array

    _plot_bars(
        cost_matrix=cost_matrix, clean_cost_array=clean_cost_array,
        predictor_names=predictor_names,
        backwards_flag=backwards_flag, multipass_flag=True,
        confidence_level=confidence_level, axes_object=axes_object,
        bar_face_colour=bar_face_colour
    )
