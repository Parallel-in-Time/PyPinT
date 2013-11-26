# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from .i_solution import ISolution
import numpy as np


class SimpleSolution(ISolution):
    """
    Summary
    -------
    Storage for the final solution of a solver.
    """
    def __init__(self):
        super(SimpleSolution, self).__init__()
        self._data = np.zeros(0, dtype=np.float64)

    def add_solution(self, data, **kwargs):
        super(SimpleSolution, self).add_solution(data, kwargs)
        self._data = data.copy()
