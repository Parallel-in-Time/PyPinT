# coding=utf-8
from pypint.plugins.function_generators import nested
import unittest
from tests.__init__ import assert_numpy_array_almost_equal
import numpy as np


# Test fields
x_1 = np.arange(10)
x_2 = np.linspace(-1, 1)
x_3 = np.array([np.pi, np.exp(-1), np.sqrt(5)])

test_fields = [x_1, x_2, x_3]


# some functions to compare
def nested_function(x):
    return np.cos(np.sqrt(x ** 6))

# Arguments to generate this functions
nested_function_list = [lambda x: x ** 6, lambda x: np.sqrt(x), lambda x: np.cos(x)]

test_options = {
    "nested_function": [
        [nested_function_list],
        nested_function
    ]
}
# End of adding cases


def correct_nested_function_generated(test_field, test_function, args):
    generator = nested.NestedFG(args[0])
    func = generator.generate_function()
    assert_numpy_array_almost_equal(func(test_field), test_function(test_field))


def test_nested_function_generator():
    for test_field in test_fields:
        for cases in test_options:
            yield correct_nested_function_generated, test_field, test_options[cases][1], \
                test_options[cases][0]


class NestedFunctionGeneratorTest(unittest.TestCase):
    pass
