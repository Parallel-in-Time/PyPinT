# coding=utf-8
import unittest

from pypint.problems.i_problem import IProblem
from pypint.problems.has_exact_solution_mixin import HasExactSolutionMixin, problem_has_exact_solution


class HasExactSolutionMixinTest(unittest.TestCase):
    class TestProblem(IProblem, HasExactSolutionMixin):
        def __init__(self, *args, **kwargs):
            super(HasExactSolutionMixinTest.TestProblem, self).__init__(*args, **kwargs)
            HasExactSolutionMixin.__init__(self, *args, **kwargs)

    def setUp(self):
        self._default = HasExactSolutionMixinTest.TestProblem()
        self._func = lambda t: 1.0

    def test_provides_exact_method(self):
        self.assertIsNone(self._default.exact_function, "Initially no exact function is given.")
        self._default.exact_function = self._func
        self.assertTrue(callable(self._default.exact_function))
        self.assertEqual(self._default.exact(0.0), 1.0)

        _test_problem = HasExactSolutionMixinTest.TestProblem(exact_function=self._func)
        self._default.exact_function = self._func
        self.assertTrue(callable(self._default.exact_function))
        self.assertEqual(self._default.exact(0.0), 1.0)

    def test_validates_exact_method_input_arguments(self):
        self._default.exact_function = self._func
        self.assertRaises(ValueError, self._default.exact, time="not a time")

    def test_problem_has_exact_solution_introspection(self):
        self.assertTrue(problem_has_exact_solution(self._default))
        self.assertFalse(problem_has_exact_solution(IProblem()))


if __name__ == '__main__':
    unittest.main()
