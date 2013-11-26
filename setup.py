#!/usr/bin/env python

import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="PyPinT",
    version="0.0.1",
    author="Torbjörn Klatt, Dieter Moser",
    author_email="t.klatt@fz-juelich.de, d.moser@fz-juelich.de",
    description=("A Python framework for Parallel-in-Time integration routines."),
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
        "Programming Language :: Python :: 3.3",
        "Topic :: Scientific/Engineering :: Mathematics"
    ]
)
