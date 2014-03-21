# coding=utf-8
from unittest.mock import patch
from nose.tools import *

import numpy

from tests import NumpyAwareTestCase, assert_numpy_array_almost_equal
from pypint.solvers.sdc import Sdc
from pypint.utilities.threshold_check import ThresholdCheck
from pypint.solvers.cores import ExplicitSdcCore, ImplicitSdcCore, SemiImplicitSdcCore
from examples.problems.lambda_u import LambdaU
from examples.problems.constant import Constant


MAX_ITER = 100
PRECISION = 6


def _run_sdc_with_problem(problem, core, num_time_steps, num_nodes, max_iter, precision):
    thresh = ThresholdCheck(max_threshold=max_iter + 1, conditions=('error', 'residual', 'solution reduction', 'error reduction', 'iterations'),
                            min_threshold=1e-7)
    _sdc = Sdc()
    _sdc.init(threshold=thresh, problem=problem, num_time_steps=num_time_steps, num_nodes=num_nodes)
    _solution = _sdc.run(core)
    # for _node_index in range(0, len(_solution.solution(-1))):
    #     print("Node {}: {} <-> {}".format(_node_index, _solution.solution(-1)[_node_index].value, problem.exact(_solution.solution(-1)[_node_index].time_point)))
        # assert_numpy_array_almost_equal(_solution.solution(-1)[_node_index].value,
        #                                 problem.exact(_solution.solution(-1)[_node_index].time_point),
        #                                 places=2)
    assert_true(len(frozenset(thresh.has_reached()).intersection(frozenset(('error', 'solution reduction', 'error reduction', 'residual')))) > 0,
                "Termination criteria should be 'error' or 'residual'.")
    assert_not_in('iterations', thresh.has_reached(), "Maximum Number of iterations should not be reached.")
    assert_numpy_array_almost_equal(_solution.solution(-1)[-1].value,
                                    problem.exact(_solution.solution(-1)[-1].time_point),
                                    places=precision)


def _constant_minus_one_function(sdc_core, num_time_steps, num_nodes, iter_precision):
    problem = Constant(constant=-1.0, shift=1.0)
    precision = iter_precision['prec'] if 'prec' in iter_precision else PRECISION
    _run_sdc_with_problem(problem, sdc_core, num_time_steps, num_nodes, iter_precision['iter'], precision)


def _lambda_u_function(sdc_core, num_time_steps, num_nodes, iter_precision):
    problem = LambdaU(lmbda=complex(-1.0, 1.0))
    precision = iter_precision['prec'] if 'prec' in iter_precision else PRECISION
    _run_sdc_with_problem(problem, sdc_core, num_time_steps, num_nodes, iter_precision['iter'], precision)


def test_constant_minus_one_function_with_explicit_sdc():
    _expected_iterations = {
        1: {3: {'iter': 1}, 5: {'iter': 1}, 7: {'iter': 1}},
        2: {3: {'iter': 1}, 5: {'iter': 1}, 7: {'iter': 1}},
        3: {3: {'iter': 1}, 5: {'iter': 1}, 7: {'iter': 1}}
    }
    for _num_time_steps in _expected_iterations.keys():
        for _num_nodes in _expected_iterations[_num_time_steps].keys():
            yield _constant_minus_one_function, ExplicitSdcCore, \
                _num_time_steps, _num_nodes, _expected_iterations[_num_time_steps][_num_nodes]


def test_constant_minus_one_function_with_implicit_sdc():
    _expected_iterations = {
        1: {3: {'iter': 1}, 5: {'iter': 1}, 7: {'iter': 1}},
        2: {3: {'iter': 1}, 5: {'iter': 1}, 7: {'iter': 1}},
        3: {3: {'iter': 1}, 5: {'iter': 1}, 7: {'iter': 1}}
    }
    for _num_time_steps in _expected_iterations.keys():
        for _num_nodes in _expected_iterations[_num_time_steps].keys():
            yield _constant_minus_one_function, ImplicitSdcCore, \
                _num_time_steps, _num_nodes, _expected_iterations[_num_time_steps][_num_nodes]


def test_constant_minus_one_function_with_semi_implicit_sdc():
    _expected_iterations = {
        1: {3: {'iter': 1}, 5: {'iter': 1}, 7: {'iter': 1}},
        2: {3: {'iter': 1}, 5: {'iter': 1}, 7: {'iter': 1}},
        3: {3: {'iter': 1}, 5: {'iter': 1}, 7: {'iter': 1}}
    }
    for _num_time_steps in _expected_iterations.keys():
        for _num_nodes in _expected_iterations[_num_time_steps].keys():
            yield _constant_minus_one_function, SemiImplicitSdcCore, \
                _num_time_steps, _num_nodes, _expected_iterations[_num_time_steps][_num_nodes]


def test_lambda_u_with_explicit_sdc():
    _expected_iterations = {
        1: {3: {'iter': 22, 'prec': 2}, 5: {'iter': 12}, 7: {'iter': 9}},
        2: {3: {'iter': 12, 'prec': 3}, 5: {'iter': 8}, 7: {'iter': 7}},
        3: {3: {'iter': 10, 'prec': 4}, 5: {'iter': 7}, 7: {'iter': 6}}
    }
    for _num_time_steps in _expected_iterations.keys():
        for _num_nodes in _expected_iterations[_num_time_steps].keys():
            yield _lambda_u_function, ExplicitSdcCore, \
                _num_time_steps, _num_nodes, _expected_iterations[_num_time_steps][_num_nodes]


def test_lambda_u_with_implicit_sdc():
    _expected_iterations = {
        1: {3: {'iter': 12, 'prec': 2}, 5: {'iter': 10}, 7: {'iter': 8}},
        2: {3: {'iter': 10, 'prec': 3}, 5: {'iter': 7}, 7: {'iter': 7}},
        3: {3: {'iter': 9, 'prec': 4}, 5: {'iter': 7}, 7: {'iter': 6}}
    }
    for _num_time_steps in _expected_iterations.keys():
        for _num_nodes in _expected_iterations[_num_time_steps].keys():
            yield _lambda_u_function, ImplicitSdcCore, \
                _num_time_steps, _num_nodes, _expected_iterations[_num_time_steps][_num_nodes]


def test_lambda_u_with_semi_implicit_sdc():
    _expected_iterations = {
        1: {3: {'iter': 14, 'prec': 2}, 5: {'iter': 11}, 7: {'iter': 9}},
        2: {3: {'iter': 10, 'prec': 3}, 5: {'iter': 8}, 7: {'iter': 7}},
        3: {3: {'iter': 9, 'prec': 4}, 5: {'iter': 7}, 7: {'iter': 6}}
    }
    for _num_time_steps in _expected_iterations.keys():
        for _num_nodes in _expected_iterations[_num_time_steps].keys():
            yield _lambda_u_function, SemiImplicitSdcCore, \
                _num_time_steps, _num_nodes, _expected_iterations[_num_time_steps][_num_nodes]


class SdcTest(NumpyAwareTestCase):
    def setUp(self):
        self._test_obj = Sdc()

    def test_sdc_solver_initialization(self):
        with patch('pypint.problems.i_initial_value_problem.IInitialValueProblem', spec=True,
                    initial_value=1.0, time_start=0.0, time_end=1.0, numeric_type=numpy.float) as IVP:
            prob = IVP.return_value
            self._test_obj.init(problem=prob, num_nodes=3)


if __name__ == "__main__":
    import unittest
    unittest.main()
