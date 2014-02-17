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


def _run_sdc_with_problem(problem, core, num_time_steps, num_nodes, max_iter):
    thresh = ThresholdCheck(max_threshold=max_iter + 1, conditions=("error", "residual", "iterations"),
                            min_threshold=1e-7)
    _sdc = Sdc()
    _sdc.init(threshold=thresh, problem=problem, num_time_steps=num_time_steps, num_nodes=num_nodes)
    _solution = _sdc.run(core)
    # for _node_index in range(0, len(_solution.solution(-1))):
    #     print("Node {}: {} <-> {}".format(_node_index, _solution.solution(-1)[_node_index].value, problem.exact(_solution.solution(-1)[_node_index].time_point)))
        # assert_numpy_array_almost_equal(_solution.solution(-1)[_node_index].value,
        #                                 problem.exact(_solution.solution(-1)[_node_index].time_point),
        #                                 places=2)
    assert_in(thresh.has_reached(), ['error', 'residual'], "Termination criteria should be 'error' or 'residual'.")
    assert_is_not(thresh.has_reached(), "iterations", "Maximum Number of iterations should not be reached.")
    assert_numpy_array_almost_equal(_solution.solution(-1)[-1].value,
                                    problem.exact(_solution.solution(-1)[-1].time_point),
                                    places=6)


def _constant_minus_one_function(sdc_core, num_time_steps, num_nodes, max_iter):
    problem = Constant(constant=-1.0, shift=1.0)
    _run_sdc_with_problem(problem, sdc_core, num_time_steps, num_nodes, max_iter)


def _lambda_u_function(sdc_core, num_time_steps, num_nodes, max_iter):
    problem = LambdaU(lmbda=complex(-1.0, -1.0))
    _run_sdc_with_problem(problem, sdc_core, num_time_steps, num_nodes, max_iter)


def test_constant_minus_one_function_with_explicit_sdc():
    _expected_iterations = {
        1: {3: 2, 5: 2, 7: 2},
        2: {3: 2, 5: 2, 7: 2},
        3: {3: 2, 5: 2, 7: 2}
    }
    for _num_time_steps in [1, 2, 3]:
        for _num_nodes in [3, 5, 7]:
            yield _constant_minus_one_function, ExplicitSdcCore, \
                _num_time_steps, _num_nodes, _expected_iterations[_num_time_steps][_num_nodes]


def test_constant_minus_one_function_with_implicit_sdc():
    _expected_iterations = {
        1: {3: 2, 5: 2, 7: 2},
        2: {3: 2, 5: 2, 7: 2},
        3: {3: 2, 5: 2, 7: 2}
    }
    for _num_time_steps in [1, 2, 3]:
        for _num_nodes in [3, 5, 7]:
            yield _constant_minus_one_function, ImplicitSdcCore, \
                _num_time_steps, _num_nodes, _expected_iterations[_num_time_steps][_num_nodes]


def test_constant_minus_one_function_with_semi_implicit_sdc():
    _expected_iterations = {
        1: {3: 2, 5: 2, 7: 2},
        2: {3: 2, 5: 2, 7: 2},
        3: {3: 2, 5: 2, 7: 2}
    }
    for _num_time_steps in [1, 2, 3]:
        for _num_nodes in [3, 5, 7]:
            yield _constant_minus_one_function, SemiImplicitSdcCore, \
                _num_time_steps, _num_nodes, _expected_iterations[_num_time_steps][_num_nodes]


def test_lambda_u_with_explicit_sdc():
    _expected_iterations = {
        2: {5: 8, 7: 7},
        3: {5: 7, 7: 7}
    }
    for _num_time_steps in [2, 3]:
        for _num_nodes in [5, 7]:
            yield _lambda_u_function, ExplicitSdcCore, \
                _num_time_steps, _num_nodes, _expected_iterations[_num_time_steps][_num_nodes]


def test_lambda_u_with_implicit_sdc():
    _expected_iterations = {
        1: {5: MAX_ITER, 7: 10},
        2: {5: MAX_ITER, 7: 9},
        3: {5: 9, 7: 9}
    }
    for _num_time_steps in [1, 2, 3]:
        for _num_nodes in [5, 7]:
            if _num_nodes == 5 and _num_time_steps in [1, 2]:
                # those do not converge
                pass
            else:
                yield _lambda_u_function, ImplicitSdcCore, \
                    _num_time_steps, _num_nodes, _expected_iterations[_num_time_steps][_num_nodes]


def test_lambda_u_with_semi_implicit_sdc():
    _expected_iterations = {
        1: {5: MAX_ITER, 7: 10},
        2: {5: 10, 7: 9},
        3: {5: 9, 7: 9}
    }
    for _num_time_steps in [1, 2, 3]:
        for _num_nodes in [5, 7]:
            if _num_time_steps == 1 and _num_nodes == 5:
                # this will not converge below error of 1e-6
                pass
            else:
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
