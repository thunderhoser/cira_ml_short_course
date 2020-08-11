"""Downloads storm data used to train and evaluate ML models.

This script is mostly copied from:

https://github.com/djgagne/ams-ml-python-course/blob/master/download_data.py
"""

import os
from urllib.request import urlretrieve
from gewittergefahr.gg_utils import unzipping
from gewittergefahr.gg_utils import file_system_utils

LOCAL_DIRECTORY_NAME = 'data'
ONLINE_TABULAR_FILE_NAME = (
    'https://storage.googleapis.com/track_data_ncar_ams_3km_csv_small/'
    'track_data_ncar_ams_3km_csv_small.tar.gz'
)
ONLINE_IMAGE_FILE_NAME = (
    'https://storage.googleapis.com/track_data_ncar_ams_3km_nc_small/'
    'track_data_ncar_ams_3km_nc_small.tar.gz'
)


def _run():
    """Downloads storm data used to train and evaluate ML models.

    This is effectively the main method.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=LOCAL_DIRECTORY_NAME
    )

    local_tabular_file_name = '{0:s}/{1:s}'.format(
        LOCAL_DIRECTORY_NAME, os.path.split(ONLINE_TABULAR_FILE_NAME)[-1]
    )
    print('Downloading data to: "{0:s}"...'.format(local_tabular_file_name))
    urlretrieve(ONLINE_TABULAR_FILE_NAME, local_tabular_file_name)

    local_image_file_name = '{0:s}/{1:s}'.format(
        LOCAL_DIRECTORY_NAME, os.path.split(ONLINE_IMAGE_FILE_NAME)[-1]
    )
    print('Downloading data to: "{0:s}"...\n'.format(local_image_file_name))
    urlretrieve(ONLINE_IMAGE_FILE_NAME, local_image_file_name)

    unzipping.unzip_tar(
        tar_file_name=local_tabular_file_name,
        target_directory_name=LOCAL_DIRECTORY_NAME
    )
    print('\n')

    unzipping.unzip_tar(
        tar_file_name=local_image_file_name,
        target_directory_name=LOCAL_DIRECTORY_NAME
    )


if __name__ == '__main__':
    _run()
