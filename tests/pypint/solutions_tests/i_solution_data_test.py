# coding=utf-8
from pypint.solutions.i_solution_data import ISolutionData
from tests import NumpyAwareTestCase
import numpy


class ISolutionDataTest(NumpyAwareTestCase):
    def setUp(self):
        self._default = ISolutionData()

    def test_provides_spacial_dimension(self):
        self.assertEqual(self._default.dim, 0, "Default spacial dimension is 0.")
        with self.assertRaises(AttributeError):
            # spacial dimension is read-only
            self._default.dim = 2

    def test_provides_numerical_type(self):
        self.assertIsNone(self._default.numeric_type, "Default numeric type is 'None'.")
        with self.assertRaises(AttributeError):
            # numeric_type is read-only
            self._default.numeric_type = numpy.float64


if __name__ == '__main__':
    import unittest
    unittest.main()
