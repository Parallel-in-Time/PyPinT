# coding=utf-8
"""

.. moduleauthor: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""


class IProblem(object):
    def __init__(self):
        self.__function = None
        self.__time_start = None
        self.__time_end = None

    def eval(self, time, phi_of_time):
        pass

    @property
    def function(self):
        return self.__function

    @function.setter
    def function(self, function):
        self.__function = function

    @property
    def time_start(self):
        return self.__time_start

    @time_start.setter
    def time_start(self, time_start):
        self.__time_start = time_start

    @property
    def time_end(self):
        return self.__time_end

    @time_end.setter
    def time_end(self, time_end):
        self.__time_end = time_end
