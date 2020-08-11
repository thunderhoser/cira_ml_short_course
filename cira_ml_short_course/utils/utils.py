"""Helper methods."""

import glob
import os.path
import time
import calendar
import numpy
import pandas
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from matplotlib import pyplot
from cira_ml_short_course.plotting import evaluation_plotting

# TODO(thunderhoser): Split this into different modules.

# Variable names.
METADATA_COLUMNS = [
    'Step_ID', 'Track_ID', 'Ensemble_Name', 'Ensemble_Member', 'Run_Date',
    'Valid_Date', 'Forecast_Hour', 'Valid_Hour_UTC'
]

EXTRANEOUS_COLUMNS = [
    'Duration', 'Centroid_Lon', 'Centroid_Lat', 'Centroid_X', 'Centroid_Y',
    'Storm_Motion_U', 'Storm_Motion_V', 'Matched', 'Max_Hail_Size',
    'Num_Matches', 'Shape', 'Location', 'Scale'
]

TARGET_NAME = 'RVORT1_MAX-future_max'
BINARIZED_TARGET_NAME = 'strong_future_rotation_flag'

MAE_KEY = 'mean_absolute_error'
RMSE_KEY = 'root_mean_squared_error'
MEAN_BIAS_KEY = 'mean_bias'
MAE_SKILL_SCORE_KEY = 'mae_skill_score'
MSE_SKILL_SCORE_KEY = 'mse_skill_score'

# Plotting constants.
FIGURE_WIDTH_INCHES = 10
FIGURE_HEIGHT_INCHES = 10

BAR_GRAPH_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
BAR_GRAPH_EDGE_WIDTH = 2
BAR_GRAPH_FONT_SIZE = 14
BAR_GRAPH_FONT_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255

FONT_SIZE = 20
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

# Misc constants.
DATE_FORMAT = '%Y%m%d'
DATE_FORMAT_REGEX = '[0-9][0-9][0-9][0-9][0-1][0-9][0-3][0-9]'

RANDOM_SEED = 6695
LAMBDA_TOLERANCE = 1e-10


def _tabular_file_name_to_date(csv_file_name):
    """Parses date from name of tabular file.

    :param csv_file_name: Path to input file.
    :return: date_string: Date (format "yyyymmdd").
    """

    pathless_file_name = os.path.split(csv_file_name)[-1]
    date_string = pathless_file_name.replace(
        'track_step_NCARSTORM_d01_', ''
    ).replace('-0000.csv', '')

    # Verify.
    time_string_to_unix(time_string=date_string, time_format=DATE_FORMAT)
    return date_string


def _remove_future_data(predictor_table):
    """Removes future data from predictors.

    :param predictor_table: pandas DataFrame with predictor values.  Each row is
        one storm object.
    :return: predictor_table: Same but with fewer columns.
    """

    predictor_names = list(predictor_table)
    columns_to_remove = [p for p in predictor_names if 'future' in p]

    return predictor_table.drop(columns_to_remove, axis=1, inplace=False)


def _lambdas_to_sklearn_inputs(lambda1, lambda2):
    """Converts lambdas to input arguments for scikit-learn.

    :param lambda1: L1-regularization weight.
    :param lambda2: L2-regularization weight.
    :return: alpha: Input arg for scikit-learn model.
    :return: l1_ratio: Input arg for scikit-learn model.
    """

    return lambda1 + lambda2, lambda1 / (lambda1 + lambda2)


def _get_reliability_curve(actual_values, predicted_values, num_bins,
                           max_bin_edge, invert=False):
    """Computes reliability curve for one target variable.

    E = number of examples
    B = number of bins

    :param actual_values: length-E numpy array of actual values.
    :param predicted_values: length-E numpy array of predicted values.
    :param num_bins: Number of bins (points in curve).
    :param max_bin_edge: Value at upper edge of last bin.
    :param invert: Boolean flag.  If True, will return inverted reliability
        curve, which bins by target value and relates target value to
        conditional mean prediction.  If False, will return normal reliability
        curve, which bins by predicted value and relates predicted value to
        conditional mean observation (target).
    :return: mean_predictions: length-B numpy array of x-coordinates.
    :return: mean_observations: length-B numpy array of y-coordinates.
    :return: example_counts: length-B numpy array with num examples in each bin.
    """

    max_bin_edge = max([max_bin_edge, numpy.finfo(float).eps])
    bin_cutoffs = numpy.linspace(0., max_bin_edge, num=num_bins + 1)

    bin_index_by_example = numpy.digitize(
        actual_values if invert else predicted_values, bin_cutoffs, right=False
    ) - 1
    bin_index_by_example[bin_index_by_example < 0] = 0
    bin_index_by_example[bin_index_by_example > num_bins - 1] = num_bins - 1

    mean_predictions = numpy.full(num_bins, numpy.nan)
    mean_observations = numpy.full(num_bins, numpy.nan)
    example_counts = numpy.full(num_bins, -1, dtype=int)

    for i in range(num_bins):
        these_example_indices = numpy.where(bin_index_by_example == i)[0]

        example_counts[i] = len(these_example_indices)
        mean_predictions[i] = numpy.mean(
            predicted_values[these_example_indices]
        )
        mean_observations[i] = numpy.mean(actual_values[these_example_indices])

    return mean_predictions, mean_observations, example_counts


def time_string_to_unix(time_string, time_format):
    """Converts time from string to Unix format.

    Unix format = seconds since 0000 UTC 1 Jan 1970.

    :param time_string: Time string.
    :param time_format: Format of time string (example: "%Y%m%d" or
        "%Y-%m-%d-%H%M%S").
    :return: unix_time_sec: Time in Unix format.
    """

    return calendar.timegm(time.strptime(time_string, time_format))


def time_unix_to_string(unix_time_sec, time_format):
    """Converts time from Unix format to string.

    Unix format = seconds since 0000 UTC 1 Jan 1970.

    :param unix_time_sec: Time in Unix format.
    :param time_format: Desired format of time string (example: "%Y%m%d" or
        "%Y-%m-%d-%H%M%S").
    :return: time_string: Time string.
    """

    return time.strftime(time_format, time.gmtime(unix_time_sec))


def find_tabular_files(directory_name, first_date_string, last_date_string):
    """Finds CSV files with tabular data.

    :param directory_name: Name of directory with tabular files.
    :param first_date_string: First date ("yyyymmdd") in range.
    :param last_date_string: Last date ("yyyymmdd") in range.
    :return: csv_file_names: 1-D list of paths to tabular files.
    """

    first_time_unix_sec = time_string_to_unix(
        time_string=first_date_string, time_format=DATE_FORMAT
    )
    last_time_unix_sec = time_string_to_unix(
        time_string=last_date_string, time_format=DATE_FORMAT
    )

    csv_file_pattern = '{0:s}/track_step_NCARSTORM_d01_{1:s}-0000.csv'.format(
        directory_name, DATE_FORMAT_REGEX
    )
    csv_file_names = glob.glob(csv_file_pattern)
    csv_file_names.sort()

    file_date_strings = [_tabular_file_name_to_date(f) for f in csv_file_names]

    file_times_unix_sec = numpy.array([
        time_string_to_unix(time_string=d, time_format=DATE_FORMAT)
        for d in file_date_strings
    ], dtype=int)

    good_indices = numpy.where(numpy.logical_and(
        file_times_unix_sec >= first_time_unix_sec,
        file_times_unix_sec <= last_time_unix_sec
    ))[0]

    return [csv_file_names[k] for k in good_indices]


def read_tabular_file(csv_file_name):
    """Reads tabular data from CSV file.

    :param csv_file_name: Path to input file.
    :return: metadata_table: pandas DataFrame with metadata.  Each row is one
        storm object.
    :return: predictor_table: pandas DataFrame with predictor values.  Each row
        is one storm object.
    :return: target_table: pandas DataFrame with target values.  Each row is one
        storm object.
    """

    predictor_table = pandas.read_csv(csv_file_name, header=0, sep=',')
    predictor_table.drop(EXTRANEOUS_COLUMNS, axis=1, inplace=True)

    metadata_table = predictor_table[METADATA_COLUMNS]
    predictor_table.drop(METADATA_COLUMNS, axis=1, inplace=True)

    target_table = predictor_table[[TARGET_NAME]]
    predictor_table.drop([TARGET_NAME], axis=1, inplace=True)
    predictor_table = _remove_future_data(predictor_table)

    return metadata_table, predictor_table, target_table


def read_many_tabular_files(csv_file_names):
    """Reads tabular data from many CSV files.

    :param csv_file_names: 1-D list of paths to input files.
    :return: metadata_table: See doc for `read_tabular_file`.
    :return: predictor_table: Same.
    :return: target_table: Same.
    """

    num_files = len(csv_file_names)
    list_of_metadata_tables = [pandas.DataFrame()] * num_files
    list_of_predictor_tables = [pandas.DataFrame()] * num_files
    list_of_target_tables = [pandas.DataFrame()] * num_files

    for i in range(num_files):
        print('Reading data from: "{0:s}"...'.format(csv_file_names[i]))

        (list_of_metadata_tables[i], list_of_predictor_tables[i],
         list_of_target_tables[i]
        ) = read_tabular_file(csv_file_names[i])

        if i == 0:
            continue

        list_of_metadata_tables[i] = list_of_metadata_tables[i].align(
            list_of_metadata_tables[0], axis=1
        )[0]

        list_of_predictor_tables[i] = list_of_predictor_tables[i].align(
            list_of_predictor_tables[0], axis=1
        )[0]

        list_of_target_tables[i] = list_of_target_tables[i].align(
            list_of_target_tables[0], axis=1
        )[0]

    metadata_table = pandas.concat(
        list_of_metadata_tables, axis=0, ignore_index=True
    )
    predictor_table = pandas.concat(
        list_of_predictor_tables, axis=0, ignore_index=True
    )
    target_table = pandas.concat(
        list_of_target_tables, axis=0, ignore_index=True
    )

    return metadata_table, predictor_table, target_table


def normalize_predictors(predictor_table, normalization_dict=None):
    """Normalizes predictors to z-scores.

    :param predictor_table: See doc for `read_tabular_file`.
    :param normalization_dict: Dictionary.  Each key is the name of a predictor
        value, and the corresponding value is a length-2 numpy array with
        [mean, standard deviation].  If `normalization_dict is None`, mean and
        standard deviation will be computed for each predictor.
    :return: predictor_table: Normalized version of input.
    :return: normalization_dict: See doc for input variable.  If input was None,
        this will be a newly created dictionary.  Otherwise, this will be the
        same dictionary passed as input.
    """

    predictor_names = list(predictor_table)
    num_predictors = len(predictor_names)

    if normalization_dict is None:
        normalization_dict = {}

        for m in range(num_predictors):
            this_mean = numpy.mean(predictor_table[predictor_names[m]].values)
            this_stdev = numpy.std(
                predictor_table[predictor_names[m]].values, ddof=1
            )

            normalization_dict[predictor_names[m]] = numpy.array(
                [this_mean, this_stdev]
            )

    for m in range(num_predictors):
        this_mean = normalization_dict[predictor_names[m]][0]
        this_stdev = normalization_dict[predictor_names[m]][1]
        these_norm_values = (
            (predictor_table[predictor_names[m]].values - this_mean) /
            this_stdev
        )

        predictor_table = predictor_table.assign(**{
            predictor_names[m]: these_norm_values
        })

    return predictor_table, normalization_dict


def denormalize_predictors(predictor_table, normalization_dict):
    """Denormalizes predictors from z-scores back to original scales.

    :param predictor_table: See doc for `normalize_predictors`.
    :param normalization_dict: Same.
    :return: predictor_table: Denormalized version of input.
    """

    predictor_names = list(predictor_table)
    num_predictors = len(predictor_names)

    for m in range(num_predictors):
        this_mean = normalization_dict[predictor_names[m]][0]
        this_stdev = normalization_dict[predictor_names[m]][1]
        these_denorm_values = (
            this_mean + this_stdev * predictor_table[predictor_names[m]].values
        )

        predictor_table = predictor_table.assign(**{
            predictor_names[m]: these_denorm_values
        })

    return predictor_table


def setup_linear_regression(lambda1=0., lambda2=0.):
    """Sets up (but does not train) linear-regression model.

    :param lambda1: L1-regularization weight.
    :param lambda2: L2-regularization weight.
    :return: model_object: Instance of `sklearn.linear_model`.
    """

    assert lambda1 >= 0
    assert lambda2 >= 0

    if lambda1 < LAMBDA_TOLERANCE and lambda2 < LAMBDA_TOLERANCE:
        return LinearRegression(fit_intercept=True, normalize=False)

    if lambda1 < LAMBDA_TOLERANCE:
        return Ridge(
            alpha=lambda2, fit_intercept=True, normalize=False,
            random_state=RANDOM_SEED
        )

    if lambda2 < LAMBDA_TOLERANCE:
        return Lasso(
            alpha=lambda1, fit_intercept=True, normalize=False,
            random_state=RANDOM_SEED
        )

    alpha, l1_ratio = _lambdas_to_sklearn_inputs(
        lambda1=lambda1, lambda2=lambda2
    )

    return ElasticNet(
        alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True, normalize=False,
        random_state=RANDOM_SEED
    )


def train_linear_regression(model_object, training_predictor_table,
                            training_target_table):
    """Trains linear-regression model.

    :param model_object: Untrained model created by `setup_linear_regression`.
    :param training_predictor_table: See doc for `read_feature_file`.
    :param training_target_table: Same.
    :return: model_object: Trained version of input.
    """

    model_object.fit(
        X=training_predictor_table.to_numpy(),
        y=training_target_table[TARGET_NAME].values
    )

    return model_object


def evaluate_regression(
        actual_values, predicted_values, mean_training_target_value,
        verbose=True, create_plots=True, dataset_name=None):
    """Evaluates regression model.

    E = number of examples

    :param actual_values: length-E numpy array of actual target values.
    :param predicted_values: length-E numpy array of predictions.
    :param mean_training_target_value: Mean target value in training data.
    :param verbose: Boolean flag.  If True, will print results to command
        window.
    :param create_plots: Boolean flag.  If True, will create plots.
    :param dataset_name: Dataset name (e.g., "validation").  Used only if
        `create_plots == True or verbose == True`.
    :return: evaluation_dict: Dictionary with the following keys.
    evaluation_dict['mean_absolute_error']: Mean absolute error (MAE).
    evaluation_dict['rmse']: Root mean squared error (RMSE).
    evaluation_dict['mean_bias']: Mean bias (signed error).
    evaluation_dict['mae_skill_score']: MAE skill score (fractional improvement
        over climatology, in range -1...1).
    evaluation_dict['mse_skill_score']: MSE skill score (fractional improvement
        over climatology, in range -1...1).
    """

    signed_errors = predicted_values - actual_values
    mean_bias = numpy.mean(signed_errors)
    mean_absolute_error = numpy.mean(numpy.absolute(signed_errors))
    rmse = numpy.sqrt(numpy.mean(signed_errors ** 2))

    climo_signed_errors = mean_training_target_value - actual_values
    climo_mae = numpy.mean(numpy.absolute(climo_signed_errors))
    climo_mse = numpy.mean(climo_signed_errors ** 2)

    mae_skill_score = (climo_mae - mean_absolute_error) / climo_mae
    mse_skill_score = (climo_mse - rmse ** 2) / climo_mse

    evaluation_dict = {
        MAE_KEY: mean_absolute_error,
        RMSE_KEY: rmse,
        MEAN_BIAS_KEY: mean_bias,
        MAE_SKILL_SCORE_KEY: mae_skill_score,
        MSE_SKILL_SCORE_KEY: mse_skill_score
    }

    if verbose or create_plots:
        dataset_name = dataset_name[0].upper() + dataset_name[1:]

    if verbose:
        print('{0:s} MAE (mean absolute error) = {1:.3e} s^-1'.format(
            dataset_name, evaluation_dict[MAE_KEY]
        ))
        print('{0:s} MSE (mean squared error) = {1:.3e} s^-2'.format(
            dataset_name, evaluation_dict[RMSE_KEY]
        ))
        print('{0:s} bias (mean signed error) = {1:.3e} s^-1'.format(
            dataset_name, evaluation_dict[MEAN_BIAS_KEY]
        ))

        message_string = (
            '{0:s} MAE skill score (improvement over climatology) = {1:.3f}'
        ).format(dataset_name, evaluation_dict[MAE_SKILL_SCORE_KEY])
        print(message_string)

        message_string = (
            '{0:s} MSE skill score (improvement over climatology) = {1:.3f}'
        ).format(dataset_name, evaluation_dict[MSE_SKILL_SCORE_KEY])
        print(message_string)

    if not create_plots:
        return evaluation_dict

    mean_predictions, mean_observations, example_counts = (
        _get_reliability_curve(
            actual_values=actual_values, predicted_values=predicted_values,
            num_bins=20, max_bin_edge=numpy.percentile(predicted_values, 99),
            invert=False
        )
    )

    inv_mean_observations, inv_example_counts = (
        _get_reliability_curve(
            actual_values=actual_values, predicted_values=predicted_values,
            num_bins=20, max_bin_edge=numpy.percentile(actual_values, 99),
            invert=True
        )[1:]
    )

    concat_values = numpy.concatenate((mean_predictions, mean_observations))

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    evaluation_plotting.plot_attributes_diagram(
        figure_object=figure_object, axes_object=axes_object,
        mean_predictions=mean_predictions, mean_observations=mean_observations,
        example_counts=example_counts,
        inv_mean_observations=inv_mean_observations,
        inv_example_counts=inv_example_counts,
        mean_value_in_training=mean_training_target_value,
        min_value_to_plot=0., max_value_to_plot=numpy.max(concat_values)
    )

    axes_object.set_xlabel(r'Forecast value (s$^{-1}$)')
    axes_object.set_ylabel(r'Conditional mean observation (s$^{-1}$)')

    title_string = '{0:s} attributes diagram for max future vorticity'.format(
        dataset_name
    )
    axes_object.set_title(title_string)
    pyplot.show()

    return evaluation_dict


def plot_model_coefficients(model_object, predictor_names):
    """Plots coefficients for linear- or logistic-regression model.

    :param model_object: Trained instance of `sklearn.linear_model`.
    :param predictor_names: 1-D list of predictor names, in the same order used
        to train the model.
    """

    coefficients = model_object.coef_
    num_dimensions = len(coefficients.shape)
    if num_dimensions > 1:
        coefficients = coefficients[0, ...]

    num_predictors = len(predictor_names)
    y_coords = numpy.linspace(
        0, num_predictors - 1, num=num_predictors, dtype=float
    )

    _, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    axes_object.barh(
        y_coords, coefficients, color=BAR_GRAPH_COLOUR,
        edgecolor=BAR_GRAPH_COLOUR, linewidth=BAR_GRAPH_EDGE_WIDTH
    )

    pyplot.xlabel('Coefficient')
    pyplot.ylabel('Predictor variable')

    pyplot.yticks([], [])
    x_tick_values, _ = pyplot.xticks()
    pyplot.xticks(x_tick_values, rotation=90)

    x_min = numpy.percentile(coefficients, 1.)
    x_max = numpy.percentile(coefficients, 99.)
    pyplot.xlim([x_min, x_max])

    for j in range(num_predictors):
        axes_object.text(
            0, y_coords[j], predictor_names[j], color=BAR_GRAPH_FONT_COLOUR,
            horizontalalignment='center', verticalalignment='center',
            fontsize=BAR_GRAPH_FONT_SIZE
        )
