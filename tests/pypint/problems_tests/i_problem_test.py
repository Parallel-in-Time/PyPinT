# coding=utf-8

from pypint.problems.i_problem import IProblem
from tests import NumpyAwareTestCase
import numpy as np


class IProblemTest(NumpyAwareTestCase):
    def setUp(self):
        self._default = IProblem()

    def test_takes_a_function(self):
        def _test_func():
            return np.pi
        _test_obj = IProblem(rhs_function_wrt_time=_test_func)
        self.assertTrue(callable(_test_obj.rhs_function_wrt_time))
        self.assertEqual(_test_obj.rhs_function_wrt_time(), np.pi)

    # def test_specifies_time_interval(self):
    #     self.assertEqual(self._default.time_start, 0.0, "Default time start is 0.0")
    #     self.assertEqual(self._default.time_end, 1.0, "Default time start is 1.0")
    #     self._default.time_start = 1.0
    #     self.assertEqual(self._default.time_start, 1.0)
    #     self._default.time_end = 2.0
    #     self.assertEqual(self._default.time_end, 2.0)
    #
    #     _test_obj = IProblem(time_start=1.0, time_end=2.0)
    #     self.assertEqual(_test_obj.time_start, 1.0)
    #     self.assertEqual(_test_obj.time_end, 2.0)

    def test_takes_a_numeric_type(self):
        _test_obj = IProblem(numeric_type=np.float)
        self.assertEqual(_test_obj.numeric_type, np.float)
        _test_obj = IProblem(numeric_type=np.complex)
        self.assertEqual(_test_obj.numeric_type, np.complex)
        self.assertRaises(ValueError, IProblem, numeric_type=bool)
        self.assertRaises(Exception, IProblem, numeric_type="not a type")

    def test_has_a_spacial_dimension(self):
        self.assertEqual(self._default.dim, 1, "Default dimension is 1.")

        _test_obj = IProblem(dim=3)
        self.assertEqual(_test_obj.dim, 3)

    def test_provides_evaluation(self):
        self._default.evaluate_wrt_time(0.0, np.array([1.0]))
        self.assertRaises(ValueError, self._default.evaluate_wrt_time, complex(1.0, 1.0), np.array([1.0]))
        self.assertRaises(ValueError, self._default.evaluate_wrt_time, 1.0, 1.0)

    def test_provides_implicit_solver(self):
        _next_x = np.array([1.0])
        _func = lambda x: 1.0 - x
        _x = self._default.implicit_solve(_next_x, _func)
        self.assertNumpyArrayAlmostEqual(_x, _next_x, places=3)
        self.assertRaises(ValueError, self._default.evaluate_wrt_time, 1.0, _func)
        self.assertRaises(ValueError, self._default.evaluate_wrt_time, _next_x, "not callable")

    def test_takes_descriptive_strings(self):
        self.assertRegex(self._default.__str__(), "IProblem")

        _test_obj = IProblem(strings={'rhs_wrt_time': "Right-Hand Side Formula"})
        self.assertRegex(_test_obj.__str__(), "Right-Hand Side Formula")


if __name__ == "__main__":
    import unittest
    unittest.main()
