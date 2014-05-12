# coding=utf-8
"""
.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
import numpy as np

from pypint.problems.i_initial_value_problem import IInitialValueProblem
from pypint.plugins.multigrid.multigrid_problem_mixin import MultigridProblemMixin
from pypint.utilities import assert_named_argument


class ITransientMultigridProblem(IInitialValueProblem, MultigridProblemMixin):
    """Interface for transient problems using multigrid as space solver
    """
    def __init__(self, *args, **kwargs):
        super(ITransientMultigridProblem, self).__init__(*args, **kwargs)
        MultigridProblemMixin.__init__(self, *args, **kwargs)

    def implicit_solve(self, next_x, func, method="hybr"):
        # TODO: the real MG-stuff for SDC solver cores goes here
        pass

    def evaluate_wrt_space(self, **kwargs):
        """
        Parameters
        ----------
        values : :py:class:`numpy.ndarray`
        """
        assert_named_argument('values', kwargs, types=np.ndarray, descriptor="Values", checking_obj=self)
        assert_named_argument('delta_time', kwargs, types=float, descriptor="Delta Time Node", checking_obj=self)
        return self.get_rhs_space_operators(kwargs['delta_time'])\
                    .dot(kwargs['values'].flatten())\
                    .reshape(kwargs['values'].shape)


__all__ = ['ITransientMultigridProblem']
