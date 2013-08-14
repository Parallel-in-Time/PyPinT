#!/usr/bin/env python

import os
from setuptools import setup

def read( fname ):
    return open( os.path.join( os.path.dirname( __file__ ), fname ) ).read()

setup( name="pySDC",
       version="0.0.1 pre-alpha",
       author="Torbjörn Klatt",
       author_email="opensource@torbjoern-klatt.de",
       description=( "A Python implementation of a Spectral Deffered Corrections (SDC) method" ),
       long_description=read( 'README' ),
       license="MIT",
       url="http://blog.torbjoern-klatt.de/",
       install_requires=['numpy'],
       packages=['pySDC', 'tests'],
       test_suite="tests",
       classifiers=[
            "Development Status :: 2 - Pre-Alpha",
            "Environment :: Console",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Natural Language :: English",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3.2",
            "Topic :: Scientific/Engineering :: Mathematics"]
 )
