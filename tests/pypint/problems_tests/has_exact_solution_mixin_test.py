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
        self._default = HasExactSolutionMixinTest.TestProblem()

    def test_provides_exact_method(self):
        self.assertIsNone(self._default.exact_function, "Initially no exact function is given.")
        _func = lambda t, x: 1.0
        self._default.exact_function = _func
        self.assertTrue(callable(self._default.exact_function))
        self.assertEqual(self._default.exact(0.0, np.array([1.0])), 1.0)

        _test_problem = HasExactSolutionMixinTest.TestProblem(exact_function=_func)
        self._default.exact_function = _func
        self.assertTrue(callable(self._default.exact_function))
        self.assertEqual(self._default.exact(0.0, np.array([1.0])), 1.0)

    def test_validates_exact_method_input_arguments(self):
        _func = lambda t, x: 1.0
        self._default.exact_function = _func
        self.assertRaises(ValueError, self._default.exact, time="not a time", phi_of_time=np.array([1.0]))
        self.assertRaises(ValueError, self._default.exact, time=1.0, phi_of_time=1.0)

    def test_problem_has_exact_solution_introspection(self):
        self.assertTrue(problem_has_exact_solution(self._default))
        self.assertFalse(problem_has_exact_solution(IProblem()))


if __name__ == '__main__':
    unittest.main()
