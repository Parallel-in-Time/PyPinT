import unittest
from decimal import Decimal

from nose.tools import raises

from pySDC.integrate.newton_cotes import NewtonCotes


testOrders = [1, 4]
testCases = {'correct': [], 'fail': []}

testCases['fail'].append(
    {'func': lambda x: 1.0, 'begin': 0, 'end': 0, 'steps': 1, 'msg': "Zero interval"})
testCases['fail'].append(
    {'func': lambda x: 1.0, 'begin': 0, 'end': 1, 'steps': 0, 'msg': "No steps"})
testCases['fail'].append(
    {'func': lambda x: 1.0, 'begin': 1, 'end': 0, 'steps': 1, 'msg': "Negative interval"})

testCases['correct'].append(
    {'func': lambda x: Decimal(0.0), 'begin': 0, 'end': 1, 'steps': 10, 'result': Decimal(0.0),
     'msg': "Zero function"})
testCases['correct'].append(
    {'func': lambda x: Decimal(1.0), 'begin': 0, 'end': 1, 'steps': 10, 'result': Decimal(1.0),
     'msg': "One function"})
testCases['correct'].append(
    {'func': lambda x: Decimal(x), 'begin': 0, 'end': 1, 'steps': 10, 'result': Decimal(0.5),
     'msg': "Identity function"})


def correct_integrate(func, begin, end, steps, order, result, message):
    assert NewtonCotes.integrate(func, begin, end, steps, order) == result, message


@raises(ValueError)
def failed_integrate(func, begin, end, steps, order, message):
    NewtonCotes.integrate(func, begin, end, steps, order)


def test_newton_cotes_integrate_correct():
    """
    """
    for order in range(testOrders[0], testOrders[1] + 1):
        for case in testCases['correct']:
            yield correct_integrate, case['func'], case['begin'], case['end'], case['steps'], \
                order, case['result'], case['msg']


def test_newton_cotes_integrate_failures():
    """
    """
    for order in range(testOrders[0], testOrders[1] + 1):
        for case in testCases['fail']:
            yield failed_integrate, case['func'], case['begin'], case['end'], case['steps'], \
                order, case['msg']


class NewtonCotesTests(unittest.TestCase):
    def test_newton_cotes_initialization(self):
        """
        """
        test_obj = NewtonCotes()
        self.assertIsInstance(test_obj, NewtonCotes)
        self.assertTrue(hasattr(test_obj, 'integrate'),
                        "Newton-Cotes integration scheme needs integrate function.")
        NewtonCotes.integrate()
        self.assertEqual(NewtonCotes.integrate(), Decimal(1.0), "Default integrate values")

    def test_newton_cotes_integrate_order_none(self):
        """
        """
        with self.assertRaises(NotImplementedError):
            NewtonCotes.integrate(order=0)


if __name__ == "__main__":
    unittest.main()
