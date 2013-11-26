# coding=utf-8

from pypint.plugins.function_generators import trigonometric
import unittest
from tests.__init__ import assert_numpy_array_almost_equal
import numpy as np


# Test fields
x_1 = np.arange(10)
x_2 = np.linspace(-1, 1)
x_3 = np.array([np.pi, np.exp(-1), np.sqrt(5)])

test_fields = [x_1, x_2, x_3]


# some functions to compare
def trig_polynom(x):
    return np.cos(5.0 * x) + np.cos(0.5 * x) + 2.0 * np.cos(2.0 * x) + np.sin(x)

# Arguments to generate this functions
trig_freqs = np.array([[5.0, 0.5, 2.0, 1.0]])
trig_coeffs = np.array([1.0, 1.0, 2.0, 1.0])
#trig_trans = np.array([[(+np.pi/2)/5, 0.0, (+np.pi/2)/2 ]])
trig_trans = np.zeros(trig_freqs.shape)
trig_trans[0, -1] = -np.pi / 2.0
test_options = {
    "trig_polynom": [
        [trig_freqs, trig_coeffs, trig_trans, None],
        trig_polynom
    ]
}
# End of adding cases


# writing generators for test cases
def correct_trigonometric_function_generated(test_field, test_function, args):
    generator = trigonometric.TrigonometricFG(args[0], args[1], args[2], args[3])
    func = generator.generate_function()
    assert_numpy_array_almost_equal(func(test_field), test_function(test_field))


def test_trigonometric_function_generator():
    for test_field in test_fields:
        for cases in test_options:
            yield correct_trigonometric_function_generated, test_field, test_options[cases][1], \
                test_options[cases][0]


class TrigonometricFunctionGeneratorTest(unittest.TestCase):
    pass
