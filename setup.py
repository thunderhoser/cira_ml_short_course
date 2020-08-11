"""Setup file for cira_ml_short_course."""

from setuptools import setup

PACKAGE_NAMES = [
    'cira_ml_short_course', 'cira_ml_short_course.utils',
    'cira_ml_short_course.plotting'
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
    'geopy'
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
