# coding=utf-8
"""Problems for Iterative Time Solvers

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
.. moduleauthor:: Dieter Moser <d.moser@fz-juelich.de>
"""

from pypint.problems.i_problem import IProblem
from pypint.problems.i_initial_value_problem import IInitialValueProblem
from pypint.problems.has_exact_solution_mixin import HasExactSolutionMixin, problem_has_exact_solution
from pypint.problems.has_direct_implicit_mixin import HasDirectImplicitMixin, problem_has_direct_implicit

__all__ = [
    'IProblem', 'IInitialValueProblem',
    'HasExactSolutionMixin', 'HasDirectImplicitMixin',
    'problem_has_exact_solution', 'problem_has_direct_implicit'
]
