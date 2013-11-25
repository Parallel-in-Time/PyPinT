# coding=utf-8
from pypint.plugins.function_generators import piecewise
import unittest
from tests.__init__ import assert_numpy_array_almost_equal
import numpy as np


# Test fields
x_1 = np.arange(10)
x_2 = np.linspace(-1, 1)
x_3 = np.array([np.pi, np.exp(-1), np.sqrt(5)])

test_fields = [x_1, x_2, x_3]


# some functions to compare
def piecewise_function(x):
    values = np.zeros(x.shape)
    for i in range(x.size):
        if x[i] <= 0:
            values[i] = 0.0
        else:
            values[i] = 1.0
    return values


# Arguments to generate this functions
piecewise_functions = [lambda x: 0.0, lambda x: 1.0]
piecewise_points = np.array([0.0])

test_options = {
    "piece_wise_function": [
        [piecewise_functions, piecewise_points],
        piecewise_function
    ]
}
# End of adding cases


# writing generators for test cases
def correct_piecewise_function_generated(test_field, test_function, args):
    generator = piecewise.PiecewiseFG(args[0], args[1])
    func = generator.generate_function()
    assert_numpy_array_almost_equal(func(test_field), test_function(test_field))


def test_piecewise_function_generator():
    for test_field in test_fields:
        for cases in test_options:
            yield correct_piecewise_function_generated, test_field, test_options[cases][1], \
                test_options[cases][0]


class PiecewiseFunctionGeneratorTest(unittest.TestCase):
    pass
