Requirements
============

It is advised to use the provided ``requirements.txt`` and ``test_requirements.txt`` for installing the required
dependencies with Pip.

Python 3.3
    *PyPinT* has been tested with Python version 3.3.
    All 2.x versions are known to not be compatible. This is on purpose.

numPy
    Required for its arrays

sciPy
    Required for its linear algebra module

matplotlib
    Required for the plotters

Sphinx (optional)
    Required to generate the documentation.
    Make sure you cloned the submodule in ``doc/sphinxext``

    sphinx-bootstrap-theme (required)
        Required for the layout and look of the generated HTML documentation.
        Please use Torbjörn's fork on GitHub as it provides a custom HTML writer.

nose (optional)
    Required for running the unit test suite.

    coverage (required)
        For test coverage analysis.
