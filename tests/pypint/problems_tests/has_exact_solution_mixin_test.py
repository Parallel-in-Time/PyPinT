# coding=utf-8

from pypint.problems.i_problem import IProblem
from pypint.problems.has_exact_solution_mixin import HasExactSolutionMixin, problem_has_exact_solution
import numpy as np
import unittest


class HasExactSolutionMixinTest(unittest.TestCase):
    class TestProblem(IProblem, HasExactSolutionMixin):
        def __init__(self, *args, **kwargs):
            super(HasExactSolutionMixinTest.TestProblem, self).__init__(*args, **kwargs)
            HasExactSolutionMixin.__init__(self, *args, **kwargs)

    def setUp(self):
        self.test_problem = HasExactSolutionMixinTest.TestProblem()

    def test_provides_exact_method(self):
        self.assertIsNone(self.test_problem.exact_function, "Initially no exact function is given.")
        _func = lambda t, x: 1.0
        self.test_problem.exact_function = _func
        self.assertTrue(callable(self.test_problem.exact_function))
        self.assertEqual(self.test_problem.exact(0.0, np.array([1.0])), 1.0)

        _test_problem = HasExactSolutionMixinTest.TestProblem(exact_function=_func)
        self.test_problem.exact_function = _func
        self.assertTrue(callable(self.test_problem.exact_function))
        self.assertEqual(self.test_problem.exact(0.0, np.array([1.0])), 1.0)

    def test_validates_exact_method_input_arguments(self):
        _func = lambda t, x: 1.0
        self.test_problem.exact_function = _func
        self.assertRaises(ValueError, self.test_problem.exact, time="not a time", phi_of_time=np.array([1.0]))
        self.assertRaises(ValueError, self.test_problem.exact, time=1.0, phi_of_time=1.0)

    def test_problem_has_exact_solution_introspection(self):
        self.assertTrue(problem_has_exact_solution(self.test_problem))
        self.assertFalse(problem_has_exact_solution(IProblem()))


if __name__ == '__main__':
    unittest.main()
