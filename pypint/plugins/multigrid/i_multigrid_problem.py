# coding=utf-8
"""
.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
from pypint.problems.i_problem import IProblem
from pypint.plugins.multigrid.multigrid_problem_mixin import MultiGridProblemMixin


class IMultigridProblem(IProblem, MultiGridProblemMixin):
    """Interface for stationary problems using multigrid as space solver
    """
    def __init__(self, *args, **kwargs):
        super(IMultigridProblem, self).__init__(*args, **kwargs)
        MultiGridProblemMixin.__init__(self, *args, **kwargs)


__all__ = ['IMultigridProblem']
