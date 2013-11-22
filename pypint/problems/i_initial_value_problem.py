# coding=utf-8
"""

.. moduleauthor: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from pypint.problems.i_problem import IProblem


class IInitialValueProblem(IProblem):
    def __init__(self, *args, **kwargs):
        super(IInitialValueProblem, self).__init__(args, kwargs)
        if "initial_value" in kwargs:
            self._initial_value = kwargs["initial_value"]
        else:
            self._initial_value = None

    @property
    def initial_value(self):
        return self._initial_value

    @initial_value.setter
    def initial_value(self, initial_value):
        self._initial_value = initial_value

    def __str__(self):
        str = super(IInitialValueProblem, self).__str__()
        str += " with initial value {:.3f}".format(self.initial_value)
        return str
