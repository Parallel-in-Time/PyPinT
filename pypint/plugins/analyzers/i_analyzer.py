# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""


class IAnalyzer(object):
    def __init__(self, *args, **kwargs):
        self._data = None
        self._plotter = None

    def run(self):
        pass

    def add_data(self, *args, **kwargs):
        pass
