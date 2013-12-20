# coding=utf-8
"""
Problems for Iterative Time Solvers

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
.. moduleauthor:: Dieter Moser <d.moser@fz-juelich.de>
"""

from .i_problem import IProblem
from .i_initial_value_problem import IInitialValueProblem
from .has_exact_mixin import HasExactSolutionMixin, problem_has_exact_solution

__all__ = ['IProblem', 'IInitialValueProblem', 'HasExactSolutionMixin', 'problem_has_exact_solution']
