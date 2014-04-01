Requirements
============

It is advised to use the provided ``requirements.txt`` (or ``requirements34.txt`` if you are on Python 3.4) and
``test_requirements.txt`` (only for testing and development) for installing the required dependencies with Pip.

`Python`_ (obviously, >= 3.3)
    *PyPinT* has been tested with Python version 3.3.x and 3.4.0.
    Versions prior to 3.2 will most likely not work as the `SciPy stack`_ is `not compatible`_ with it.
    All 2.x versions are known to not be compatible. This is on purpose. [#]_

`enum34`_ (required if Python < 3.4)
    Enumeration types have been introduced by Python 3.4 and are backported as this package.

`NumPy`_ (required, >= 1.6.1)
    Required for its arrays

`SciPy`_ (required, >= 0.9.0)
    Required for its linear algebra module

`matplotlib`_ (required, >= 1.2.0)
    Required for the plotters

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

.. _Python: http://python.org/
.. _SciPy stack: http://www.scipy.org/install.html
.. _not compatible: http://www.scipy.org/stackspec.html
.. _enum34: https://pypi.python.org/pypi/enum34
.. _NumPy: http://numpy.scipy.org/
.. _SciPy: http://www.scipy.org/scipylib/index.html
.. _matplotlib: http://matplotlib.org/
.. _Logbook: https://pythonhosted.org/Logbook/index.html
.. _Sphinx: http://sphinx-doc.org/
.. _sphinx-rtd-theme: https://github.com/snide/sphinx_rtd_theme
.. _nose: https://nose.readthedocs.org/en/latest/
.. _coverage: https://pypi.python.org/pypi/coverage
.. _nose-cover3: https://pypi.python.org/pypi/nose-cover3

.. rubric:: Footnotes

.. [#] Blame your system's administrator in case you don't have any Python 3 available.
