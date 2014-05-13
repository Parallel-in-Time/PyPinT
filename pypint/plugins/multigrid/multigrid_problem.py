# coding=utf-8
"""
.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
from pypint.plugins.multigrid.i_multigrid_problem import IMultigridProblem


class MultigridProblem(IMultigridProblem):
    def __init__(self, *args, **kwargs):
        super(MultigridProblem, self).__init__(*args, **kwargs)


__all__ = ['MultigridProblem']
