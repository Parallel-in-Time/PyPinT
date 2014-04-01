#!/usr/bin/env python

import os
from setuptools import setup

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
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="PyPinT",
    version=verstr,
    author="Torbjörn Klatt, Dieter Moser",
    author_email="t.klatt@fz-juelich.de, d.moser@fz-juelich.de",
    description="A Python framework for Parallel-in-Time integration routines.",
    long_description=read('README.md'),
    license="MIT",
    url="http://fz-juelich.de/ias/jsc/pint",
    install_requires=['numpy', 'scipy', 'matplotlib'],
    packages=['pypint', 'tests'],
    test_suite="tests",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.2",
        "Topic :: Scientific/Engineering :: Mathematics"
    ]
)
