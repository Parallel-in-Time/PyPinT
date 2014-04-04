#!/usr/bin/env python

# credit, where credit due:
# A SIGNIFICANT PART OF THIS FILE IS TAKEN FROM numpy/setup.py

import os
from setuptools import setup, find_packages
from sys import version_info

if version_info.major < 3 and version_info.minor < 3:
    raise RuntimeError("PyPinT requires at least Python 3.3.")

import subprocess
import builtins


MAJOR = 0
MINOR = 0
MICRO = 3
ISRELEASED = True
RELEASE_CANDIDATE = 2  # set to None or 0 if it's not a release candidate

VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)
if RELEASE_CANDIDATE and RELEASE_CANDIDATE > 0:
    VERSION += '-rc%d' % RELEASE_CANDIDATE


def git_version():
    """Taken from numpy/setup.py
    """
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        git_revision = out.strip().decode('ascii')
    except OSError:
        git_revision = "Unknown"

    try:
        out = _minimal_ext_cmd(['git', 'describe', '--abbrev=0'])
        git_tag = out.strip().decode('ascii')
    except OSError:
        git_tag = "Unknown"

    if git_tag != "Unknown":
        try:
            out = _minimal_ext_cmd(['git', 'rev-list', '%s..HEAD' % git_tag, '--count'])
            git_commits_since = out.strip().decode('ascii')
        except OSError:
            git_commits_since = None
    else:
        git_commits_since = None

    try:
        out = _minimal_ext_cmd(['git', 'describe'])
        git_desc = out.strip().decode('ascii')
    except OSError:
        git_desc = "Unknown"

    return git_revision, git_tag, git_commits_since, git_desc


# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly update it when the contents of directories
# change.
if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')


# This is a bit hackish: we are setting a global variable so that the main pypint __init__ can detect if it is being
# loaded by the setup routine, to avoid attempting to load components that aren't built yet. While ugly, it's a lot
# more robust than what was previously being used.
builtins.__PYPINT_SETUP__ = True


def get_version_info():
    """Taken from numpy/setup.py
    """
    # Adding the git rev number needs to be done inside write_version_py(), otherwise the import of pypint.version
    # messes up the build under Python 3.
    fullversion = VERSION
    if os.path.exists('.git'):
        git_revision, git_tag, git_commits_since, git_desc = git_version()
    elif os.path.exists('pypint/version.py'):
        # must be a source distribution, use existing version file
        try:
            from pypint.version import git_revision
            from pypint.version import git_tag
            from pypint.version import git_commits_since
            from pypint.version import git_desc
        except ImportError:
            raise ImportError("Unable to import git_revision. Try removing pypint/version.py and the build directory "
                              "before building.")
    else:
        git_revision = "Unknown"
        git_tag = "Unknown"
        git_commits_since = None
        git_desc = "Unknown"

    if not ISRELEASED:
        fullversion += '-dev'
        if git_commits_since:
            fullversion += '-' + git_commits_since
        fullversion += '-g' + git_revision[:7]

    return fullversion, git_revision, git_tag, git_commits_since, git_desc


def write_version_py(filename='pypint/version.py'):
    """Taken from numpy/setup.py
    """
    cnt = """# coding=utf-8
# THIS FILE IS GENERATED FROM PYPINT SETUP.PY
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
git_revision = '%(git_revision)s'
git_tag = '%(git_tag)s'
git_commits_since = '%(git_commits_since)s'
git_desc = '%(git_desc)s'
release = %(isrelease)s

if not release:
    version = full_version
"""
    full_version, git_revision, git_tag, git_commits_since, git_desc = get_version_info()

    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION,
                       'full_version': full_version,
                       'git_revision': git_revision,
                       'git_tag': git_tag,
                       'git_commits_since': git_commits_since,
                       'git_desc': git_desc,
                       'isrelease': str(ISRELEASED)})
    finally:
        a.close()


# rewrite the pypint/version.py each time
write_version_py()


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
    'version': get_version_info()[0],
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
