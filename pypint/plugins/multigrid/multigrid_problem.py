# coding=utf-8
"""
.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
from pypint.problems.i_problem import IProblem
from pypint.plugins.multigrid.i_multigrid_problem import IMultigridProblem


class MultigridProblem(IProblem, IMultigridProblem):
    pass


__all__ = ['MultigridProblem']
