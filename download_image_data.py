"""Downloads image data for ML models.

This script is mostly copied from:

https://github.com/djgagne/ams-ml-python-course/blob/master/download_data.py
"""

import os
import errno
import tarfile
from urllib.request import urlretrieve

LOCAL_DIRECTORY_NAME = 'data'
ONLINE_IMAGE_FILE_NAME = (
    'https://storage.googleapis.com/track_data_ncar_ams_3km_nc_small/'
    'track_data_ncar_ams_3km_nc_small.tar.gz'
)


def _mkdir_recursive_if_necessary(directory_name=None, file_name=None):
    """Creates directory if necessary (i.e., doesn't already exist).

    This method checks for the argument `directory_name` first.  If
    `directory_name` is None, this method checks for `file_name` and extracts
    the directory.

    :param directory_name: Path to local directory.
    :param file_name: Path to local file.
    """

    if directory_name is None:
        assert isinstance(file_name, str)
        directory_name = os.path.dirname(file_name)
    else:
        assert isinstance(directory_name, str)

    if directory_name == '':
        return

    try:
        os.makedirs(directory_name)
    except OSError as this_error:
        if this_error.errno == errno.EEXIST and os.path.isdir(directory_name):
            pass
        else:
            raise


def _run():
    """Downloads storm data used to train and evaluate ML models.

    This is effectively the main method.
    """

    _mkdir_recursive_if_necessary(directory_name=LOCAL_DIRECTORY_NAME)

    local_image_file_name = '{0:s}/{1:s}'.format(
        LOCAL_DIRECTORY_NAME, os.path.split(ONLINE_IMAGE_FILE_NAME)[-1]
    )
    print('Downloading data to: "{0:s}"...'.format(local_image_file_name))
    urlretrieve(ONLINE_IMAGE_FILE_NAME, local_image_file_name)

    print('Unzipping file: "{0:s}"...'.format(local_image_file_name))
    tar_file_handle = tarfile.open(local_image_file_name)
    tar_file_handle.extractall(LOCAL_DIRECTORY_NAME)
    tar_file_handle.close()


if __name__ == '__main__':
    _run()
