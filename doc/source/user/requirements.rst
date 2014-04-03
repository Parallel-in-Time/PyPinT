Requirements
============

It is advised to use `setuptools`_ via the provided ``setup.py`` file for pulling in the required dependencies.
The dependencies for building the documentation and running the test coverage analysis, two requirements files to be
passed to ``pip install -r`` are provided: ``test_requirements.txt`` and ``docu_requirements.txt``.

`Python`_ (obviously, >= 3.3)
    *PyPinT* has been tested with Python version 3.3.x and 3.4.0.
    Python 3.2 does not have :py:mod:`unittest.mock` included [#]_ and do not support Unicode literals of the form
    ``u'text`` (cf. `PEP414`_), which is used by `Logbook`_, thus Python 3.2 is not supported.
    All 2.x versions are known to not be compatible. This is on purpose. [#]_

`enum34`_ (required if Python < 3.4)
    Enumeration types have been introduced by Python 3.4 and are backported as this package.

`NumPy`_ (required, >= 1.6.1)
    Required for its arrays

`SciPy`_ (required, >= 0.9.0)
    Required for its linear algebra module

`matplotlib`_ (required, >= 1.2.0)
    Required for the plotters

`configobj`_ (required, >=5.0.2)
    Required for dealing with configuration files

`Logbook`_ (required, >= 0.6.0)
    The logging output is handeled via *LogBook*.

`Sphinx`_ (optional, >= 1.3a0)
    Required to generate the documentation.
    See :doc:`../development/documentation` for further details.

    `sphinx-rtd-theme`_ (required)
        Required for the layout and look of the generated HTML documentation.

`nose`_ (optional, >= 1.3.1)
    Required for running the unit test suite.
    See :doc:`../development/testing` for further details.

    `coverage`_ (strongly suggested)
        For test coverage analysis.

    `nose-cover3`_ (required if *coverage* is installed)
        Plugin for nose to generate test coverage.

.. _setuptools: https://pypi.python.org/pypi/setuptools
.. _Python: http://python.org/
.. _PEP414: http://www.python.org/dev/peps/pep-0414
.. _SciPy stack: http://www.scipy.org/install.html
.. _not compatible: http://www.scipy.org/stackspec.html
.. _enum34: https://pypi.python.org/pypi/enum34
.. _NumPy: http://numpy.scipy.org/
.. _SciPy: http://www.scipy.org/scipylib/index.html
.. _matplotlib: http://matplotlib.org/
.. _configobj: https://github.com/DiffSK/configobj
.. _Logbook: https://pythonhosted.org/Logbook/index.html
.. _Sphinx: http://sphinx-doc.org/
.. _sphinx-rtd-theme: https://github.com/snide/sphinx_rtd_theme
.. _nose: https://nose.readthedocs.org/en/latest/
.. _coverage: https://pypi.python.org/pypi/coverage
.. _nose-cover3: https://pypi.python.org/pypi/nose-cover3

.. rubric:: Footnotes

.. [#] https://docs.python.org/3.3/library/unittest.mock.html#module-unittest.mock
.. [#] Blame your system's administrator in case you don't have any Python 3 available.
