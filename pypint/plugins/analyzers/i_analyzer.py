# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""


class IAnalyzer(object):
    """
    Summary
    -------
    Basic interface for analyzers.
    """
    def __init__(self, *args, **kwargs):
        self._data = None
        self._plotter = None

    def run(self):
        """
        Summary
        -------
        Runs the full analyzation procedure of this analyzer.
        """
        pass

    def add_data(self, *args, **kwargs):
        """
        Summary
        -------
        Adds a dataset to the analyzer.
        """
        pass
