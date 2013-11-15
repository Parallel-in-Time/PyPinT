# coding=utf-8
"""

.. moduleauthor: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from pypint.problems.i_problem import IProblem


class IInitialValueProblem(IProblem):
    def __init__(self):
        self.__initial_value = None

    @property
    def initial_value(self):
        return self.__initial_value

    @initial_value.setter
    def initial_value(self, initial_value):
        self.__initial_value = initial_value
