"""Helper methods for image data."""

import copy
import glob
import h5py
import os.path
import numpy
import netCDF4
from cira_ml_short_course.utils import utils

DATE_FORMAT = '%Y%m%d'
DATE_FORMAT_REGEX = '[0-9][0-9][0-9][0-9][0-1][0-9][0-3][0-9]'

REFLECTIVITY_NAME = 'reflectivity_dbz'
TEMPERATURE_NAME = 'temperature_kelvins'
U_WIND_NAME = 'u_wind_m_s01'
V_WIND_NAME = 'v_wind_m_s01'
TARGET_NAME = 'max_future_vorticity_s01'

PREDICTOR_NAMES = [
    REFLECTIVITY_NAME, TEMPERATURE_NAME, U_WIND_NAME, V_WIND_NAME
]

PREDICTOR_NAME_TO_ORIG = {
    REFLECTIVITY_NAME: 'REFL_COM_curr',
    TEMPERATURE_NAME: 'T2_curr',
    U_WIND_NAME: 'U10_curr',
    V_WIND_NAME: 'V10_curr'
}

STORM_ID_NAME_ORIG = 'track_id'
STORM_STEP_NAME_ORIG = 'track_step'
TARGET_NAME_ORIG = 'RVORT1_MAX_future'

STORM_IDS_KEY = 'storm_id_nums'
STORM_STEPS_KEY = 'storm_steps'
PREDICTOR_NAMES_KEY = 'predictor_names'
PREDICTOR_MATRIX_KEY = 'predictor_matrix'
TARGET_NAME_KEY = 'target_name'
TARGET_MATRIX_KEY = 'target_matrix'


def __rescue_netcdf3_file(netcdf3_file_name):
    """Rescues NetCDF3 file by reading it with h5py.

    This is necessary if the NetCDF3 file is old enough that it is no longer
    compatible with the modern version of the `netCDF4` library.

    :param netcdf3_file_name: Path to input file.
    :return: dataset_object: Instance of `netCDF4.Dataset`.
    """

    dataset_object = netCDF4.Dataset(
        'in_memory_dataset', mode='w', diskless=True, persist=True
    )

    with h5py.File(netcdf3_file_name, 'r') as hdf5_file_handle:

        # First, copy dimensions to `netCDF4.Dataset` object.
        for variable_name, variable_record in hdf5_file_handle.items():
            if 'DIMENSION_LIST' in variable_record.attrs:
                continue

            dimension_length = (
                variable_record.shape[0] if len(variable_record.shape) > 0
                else 1
            )
            dataset_object.createDimension(variable_name, dimension_length)

        # Next, copy variables to `netCDF4.Dataset` object.
        for variable_name, variable_record in hdf5_file_handle.items():
            if 'DIMENSION_LIST' not in variable_record.attrs:
                continue

            if (
                    variable_name == 'time' or
                    variable_name.startswith('track_') or
                    variable_name.startswith('centroid_')
            ):
                dimension_names = ('p',)
            else:
                dimension_names = ('p', 'row', 'col')

            new_variable_record = dataset_object.createVariable(
                variable_name,
                datatype=variable_record.dtype,
                dimensions=dimension_names
            )
            new_variable_record[:] = variable_record[:]

            for attribute_name, attribute_value in variable_record.attrs.items():
                if attribute_name == 'DIMENSION_LIST':
                    continue

                new_variable_record.setncattr(attribute_name, attribute_value)

        # Finally, copy global attributes.
        for attribute_name, attribute_value in hdf5_file_handle.attrs.items():
            if attribute_name == '_NCProperties':
                continue

            dataset_object.setncattr(attribute_name, attribute_value)

    return dataset_object


def _file_name_to_date(netcdf_file_name):
    """Parses date from name of image (NetCDF) file.

    :param netcdf_file_name: Path to input file.
    :return: date_string: Date (format "yyyymmdd").
    """

    pathless_file_name = os.path.split(netcdf_file_name)[-1]

    date_string = pathless_file_name.replace(
        'NCARSTORM_', ''
    ).replace('-0000_d01_model_patches.nc', '')

    # Verify.
    utils.time_string_to_unix(time_string=date_string, time_format=DATE_FORMAT)
    return date_string


def find_many_files(directory_name, first_date_string, last_date_string):
    """Finds image (NetCDF) files in the given date range.

    :param directory_name: Name of directory with image (NetCDF) files.
    :param first_date_string: First date ("yyyymmdd") in range.
    :param last_date_string: Last date ("yyyymmdd") in range.
    :return: netcdf_file_names: 1-D list of paths to image files.
    """

    first_time_unix_sec = utils.time_string_to_unix(
        time_string=first_date_string, time_format=DATE_FORMAT
    )
    last_time_unix_sec = utils.time_string_to_unix(
        time_string=last_date_string, time_format=DATE_FORMAT
    )

    netcdf_file_pattern = (
        '{0:s}/NCARSTORM_{1:s}-0000_d01_model_patches.nc'
    ).format(directory_name, DATE_FORMAT_REGEX)

    netcdf_file_names = glob.glob(netcdf_file_pattern)
    netcdf_file_names.sort()

    file_date_strings = [_file_name_to_date(f) for f in netcdf_file_names]
    file_times_unix_sec = numpy.array([
        utils.time_string_to_unix(time_string=d, time_format=DATE_FORMAT)
        for d in file_date_strings
    ], dtype=int)

    good_indices = numpy.where(numpy.logical_and(
        file_times_unix_sec >= first_time_unix_sec,
        file_times_unix_sec <= last_time_unix_sec
    ))[0]

    return [netcdf_file_names[k] for k in good_indices]


def read_file(netcdf_file_name):
    """Reads storm-centered images from NetCDF file.

    E = number of examples (storm objects) in file
    M = number of rows in each storm-centered grid
    N = number of columns in each storm-centered grid
    C = number of channels (predictor variables)

    :param netcdf_file_name: Path to input file.
    :return: image_dict: Dictionary with the following keys.
    image_dict['storm_id_nums']: length-E list of storm IDs (integers).
    image_dict['storm_steps']: length-E numpy array of storm steps (integers).
    image_dict['predictor_names']: length-C list of predictor names.
    image_dict['predictor_matrix']: E-by-M-by-N-by-C numpy array of predictor
        values.
    image_dict['target_name']: Name of target variable.
    image_dict['target_matrix']: E-by-M-by-N numpy array of target values.
    """

    try:
        dataset_object = netCDF4.Dataset(netcdf_file_name)
    except:
        dataset_object = __rescue_netcdf3_file(netcdf_file_name)

    storm_id_nums = numpy.array(
        dataset_object.variables[STORM_ID_NAME_ORIG][:], dtype=int
    )
    storm_steps = numpy.array(
        dataset_object.variables[STORM_STEP_NAME_ORIG][:], dtype=int
    )

    predictor_matrix = None

    for this_predictor_name in PREDICTOR_NAMES:
        this_key = PREDICTOR_NAME_TO_ORIG[this_predictor_name]

        this_predictor_matrix = numpy.array(
            dataset_object.variables[this_key][:], dtype=float
        )
        this_predictor_matrix = numpy.expand_dims(
            this_predictor_matrix, axis=-1
        )

        if predictor_matrix is None:
            predictor_matrix = this_predictor_matrix + 0.
        else:
            predictor_matrix = numpy.concatenate(
                (predictor_matrix, this_predictor_matrix), axis=-1
            )

    target_matrix = numpy.array(
        dataset_object.variables[TARGET_NAME_ORIG][:], dtype=float
    )

    return {
        STORM_IDS_KEY: storm_id_nums,
        STORM_STEPS_KEY: storm_steps,
        PREDICTOR_NAMES_KEY: PREDICTOR_NAMES,
        PREDICTOR_MATRIX_KEY: predictor_matrix,
        TARGET_NAME_KEY: TARGET_NAME,
        TARGET_MATRIX_KEY: target_matrix
    }


def read_many_files(netcdf_file_names):
    """Reads storm-centered images from many NetCDF files.

    :param netcdf_file_names: 1-D list of paths to input files.
    :return: image_dict: See doc for `read_file`.
    """

    image_dict = None
    keys_to_concat = [
        STORM_IDS_KEY, STORM_STEPS_KEY, PREDICTOR_MATRIX_KEY, TARGET_MATRIX_KEY
    ]

    for this_file_name in netcdf_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_image_dict = read_file(this_file_name)

        if image_dict is None:
            image_dict = copy.deepcopy(this_image_dict)
            continue

        for this_key in keys_to_concat:
            image_dict[this_key] = numpy.concatenate((
                image_dict[this_key], this_image_dict[this_key]
            ), axis=0)

    return image_dict
