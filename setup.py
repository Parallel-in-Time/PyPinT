#!/usr/bin/env python

import os
from setuptools import setup, find_packages
from sys import version_info

##
# this block is taken from
# http://stackoverflow.com/a/7071358
import re
VERSIONFILE = "pypint/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % VERSIONFILE)
##


def read(fname):
    return open(os.path.join(os.path.abspath(os.path.dirname(__file__)), fname)).read()


CLASSIFIERS = [
    "Development Status :: 2 - Pre-Alpha",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Natural Language :: English",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3.3",
    "Programming Language :: Python :: 3.4",
    "Topic :: Scientific/Engineering :: Mathematics"
]

DEPENDENCIES = [
    "numpy>=1.6.1",
    "scipy>=0.9.0",
    "matplotlib>=1.2.0",
    "logbook>=0.6.0",
    "configobj>=5.0.2"
]

if version_info.major == 3 and version_info.minor < 4:
    DEPENDENCIES.append("enum34")

TEST_DEPENDENCIES = [
    "nose>=1.3.1"
]

metadata = {
    'name': "PyPinT",
    'version': verstr,
    'packages': find_packages(),

    'install_requires': DEPENDENCIES,
    'include_package_data': True,

    'test_suite': "tests",
    'tests_require': TEST_DEPENDENCIES,

    'author': "Torbjörn Klatt, Dieter Moser",
    'author_email': "t.klatt@fz-juelich.de, d.moser@fz-juelich.de",
    'description': "a Python framework for Parallel-in-Time integration routines",
    'long_description': read('README.md'),
    'license': "MIT",
    'url': "https://github.com/torbjoernk/PyPinT",

    'classifiers': CLASSIFIERS
}

setup(**metadata)
