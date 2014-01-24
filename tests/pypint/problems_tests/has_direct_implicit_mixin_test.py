# coding=utf-8

from pypint.problems.i_problem import IProblem
from pypint.problems.has_direct_implicit_mixin import HasDirectImplicitMixin, problem_has_direct_implicit
import unittest


class HasDirectImplicitMixinTest(unittest.TestCase):
    class TestProblem(IProblem, HasDirectImplicitMixin):
        def __init__(self, *args, **kwargs):
            super(HasDirectImplicitMixinTest.TestProblem, self).__init__(*args, **kwargs)
            HasDirectImplicitMixin.__init__(self, *args, **kwargs)

    def setUp(self):
        self.test_problem = HasDirectImplicitMixinTest.TestProblem()

    def test_provides_direct_implicit_method(self):
        self.assertRaises(NotImplementedError, self.test_problem.direct_implicit)

    def test_problem_has_direct_implicit_introspection(self):
        self.assertTrue(problem_has_direct_implicit(self.test_problem))
        self.assertFalse(problem_has_direct_implicit(IProblem()))


if __name__ == '__main__':
    unittest.main()
