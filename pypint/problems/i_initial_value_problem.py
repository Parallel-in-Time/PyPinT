# coding=utf-8
"""

.. moduleauthor: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from pypint.problems.i_problem import IProblem


class IInitialValueProblem(IProblem):
    def __init__(self):
        self._initial_value = None

    @property
    def initial_value(self):
        return self._initial_value

    @initial_value.setter
    def initial_value(self, initial_value):
        self._initial_value = initial_value
