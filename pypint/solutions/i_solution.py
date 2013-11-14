# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""


class ISolution(object):
    def __init__(self):
        self.__data = None
        self.__used_iterations = None
        self.__reduction = None

    @property
    def data(self):
        return self.__data

    @data.setter
    def data(self, data):
        self.__data = data

    @property
    def used_iterations(self):
        return self.__used_iterations

    @used_iterations.setter
    def used_iterations(self, used_iterations):
        self.__used_iterations = used_iterations

    @property
    def reduction(self):
        return self.__reduction

    @reduction.setter
    def reduction(self, reduction):
        self.__reduction = reduction
