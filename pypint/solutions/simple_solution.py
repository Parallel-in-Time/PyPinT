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
    def __init__(self, numeric_type=np.float):
        super(SimpleSolution, self).__init__(numeric_type)
        self._data = np.zeros(0, dtype=self.numeric_type)

    def add_solution(self, data, **kwargs):
        super(SimpleSolution, self).add_solution(data, kwargs)
        self._data = data.copy()
