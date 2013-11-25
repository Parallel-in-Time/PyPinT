# coding=utf-8

import unittest
from nose.tools import *
import numpy


class PyPintTestSuite(unittest.TestSuite):
    def __init__(self):
        pass


def assert_numpy_array_almost_equal(first, second, places=None, delta=None):
    assert_is_instance(first, numpy.ndarray, "First argument is not a numpy array")
    assert_is_instance(second, numpy.ndarray, "Second argument is not a numpy array")
    assert_tuple_equal(first.shape, second.shape, "The two arrays have different shape")
    first_flat = first.flatten()
    second_flat = second.flatten()
    assert_equal(first_flat.size, second_flat.size, "The two arrays are of different size")

    for index in range(0, first_flat.size):
        assert_almost_equal(first_flat[index], second_flat[index], places=places, delta=delta,
                            msg="Element {:d} not equal: {:f} != {:f}"
                                .format(index, first_flat[index], second_flat[index]))


class NumpyAwareTestCase(unittest.TestCase):
    def assertNumpyArrayAlmostEqual(self, first, second, places=None, delta=None):
        self.assertIsInstance(first, numpy.ndarray, "First argument is not a numpy array")
        self.assertIsInstance(second, numpy.ndarray, "Second argument is not a numpy array")
        self.assertTupleEqual(first.shape, second.shape, "The two arrays have different shape")
        first_flat = first.flatten()
        second_flat = second.flatten()
        self.assertEqual(first_flat.size, second_flat.size, "The two arrays are of different size")

        for index in range(0, first.size):
            self.assertAlmostEqual(first_flat[index], second_flat[index],
                                   places=places, delta=delta,
                                   msg="Element {:d} not equal: {:f} != {:f}"
                                       .format(index, first_flat[index], second_flat[index]))


if __name__ == "__main__":
    unittest.main()
