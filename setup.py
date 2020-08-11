"""Setup file for cira_ml_short_course."""

from setuptools import setup

PACKAGE_NAMES = [
    'cira_ml_short_course', 'cira_ml_short_course.utils'
]
KEYWORDS = [
    'machine learning', 'deep learning', 'artificial intelligence',
    'data science', 'weather', 'meteorology', 'thunderstorm', 'wind', 'tornado'
]
SHORT_DESCRIPTION = (
    'Notebooks for CIRA (Cooperative Institute for Research in the Atmosphere) '
    'machine-learning short course.'
)
LONG_DESCRIPTION = SHORT_DESCRIPTION
CLASSIFIERS = [
    'Development Status :: 2 - Pre-Alpha',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python :: 3'
]

# You also need to install the following packages, which are not available in
# pip.  They can both be installed by "git clone" and "python setup.py install",
# the normal way one installs a GitHub package.
#
# https://github.com/matplotlib/basemap
# https://github.com/tkrajina/srtm.py

PACKAGE_REQUIREMENTS = [
    'numpy',
    'scipy',
    'tensorflow',
    'keras',
    'scikit-learn',
    'scikit-image',
    'netCDF4',
    'pyproj',
    'opencv-python',
    'matplotlib',
    'pandas',
    'shapely',
    'descartes',
    'geopy',
    'metpy'
]

if __name__ == '__main__':
    setup(
        name='cira_ml_short_course',
        version='0.1',
        description=SHORT_DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        author='Ryan Lagerquist',
        author_email='ralager@colostate.edu',
        url='https://github.com/thunderhoser/cira_ml_short_course',
        packages=PACKAGE_NAMES,
        scripts=[],
        keywords=KEYWORDS,
        classifiers=CLASSIFIERS,
        include_package_data=True,
        zip_safe=False,
        install_requires=PACKAGE_REQUIREMENTS
    )
