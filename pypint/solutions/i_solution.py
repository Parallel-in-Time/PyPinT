# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""


class ISolution(object):
    def __init__(self):
        self._data = None
        self._used_iterations = None
        self._reduction = None

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    @property
    def used_iterations(self):
        return self._used_iterations

    @used_iterations.setter
    def used_iterations(self, used_iterations):
        self._used_iterations = used_iterations

    @property
    def reduction(self):
        return self._reduction

    @reduction.setter
    def reduction(self, reduction):
        self._reduction = reduction
