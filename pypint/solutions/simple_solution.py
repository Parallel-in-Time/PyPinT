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

    def add_solution(self, points, values, *args, **kwargs):
        super(SimpleSolution, self).add_solution(points, values, *args, **kwargs)
        _values = np.array(values, dtype=self.numeric_type)
        _errors = np.array(kwargs["error"], dtype=self.numeric_type) if "error" in kwargs else None
        _residuals = np.array(kwargs["residual"], dtype=self.numeric_type) if "residual" in kwargs else None
        self._data.init(iteration=-1, values=_values, errors=_errors, residuals=_residuals,
                                   numeric_type=self.numeric_type)
