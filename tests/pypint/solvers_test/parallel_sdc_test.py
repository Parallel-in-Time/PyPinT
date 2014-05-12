# coding=utf-8
from nose.tools import *

from tests import NumpyAwareTestCase
from pypint.integrators.sdc_integrator import SdcIntegrator
from pypint.solvers.parallel_sdc import ParallelSdc
from pypint.communicators.forward_sending_messaging import ForwardSendingMessaging
from pypint.utilities.threshold_check import ThresholdCheck
from pypint.solvers.cores import ExplicitSdcCore, ImplicitSdcCore, SemiImplicitSdcCore
from examples.problems.lambda_u import LambdaU
from examples.problems.constant import Constant


MAX_ITER = 100
PRECISION = 6


def _run_sdc_with_problem(problem, core, num_time_steps, dt, num_nodes, max_iter, precision):
    thresh = ThresholdCheck(max_threshold=max_iter + 1, conditions=('error', 'residual', 'solution reduction', 'error reduction', 'iterations'),
                            min_threshold=1e-7)
    _comm = ForwardSendingMessaging()
    _sdc = ParallelSdc(communicator=_comm)
    _comm.link_solvers(previous=_comm, next=_comm)
    _comm.write_buffer(value=problem.initial_value, time_point=problem.time_start)
    _sdc.init(integrator=SdcIntegrator, threshold=thresh, problem=problem, num_time_steps=num_time_steps, num_nodes=num_nodes)
    _solution = _sdc.run(core, dt=dt)
    # for _node_index in range(0, len(_solution.solution(-1))):
    #     print("Node {}: {} <-> {}".format(_node_index, _solution.solution(-1)[_node_index].value, problem.exact(_solution.solution(-1)[_node_index].time_point)))
        # assert_numpy_array_almost_equal(_solution.solution(-1)[_node_index].value,
        #                                 problem.exact(_solution.solution(-1)[_node_index].time_point),
        #                                 places=2)
    assert_true(len(frozenset(thresh.has_reached()).intersection(frozenset(('error', 'solution reduction', 'error reduction', 'residual')))) > 0,
                "Termination criteria should be 'error' or 'residual'.")
    assert_not_in('iterations', thresh.has_reached(), "Maximum Number of iterations should not be reached.")
    # TODO: get the solution output of ParallelSDC right
    # assert_numpy_array_almost_equal(_solution[-1].solution(-1)[-1].value,
    #                                 problem.exact(_solution[-1].solution(-1)[-1].time_point),
    #                                 places=precision)


def _constant_minus_one_function(sdc_core, num_time_steps, dt, num_nodes, iter_precision):
    problem = Constant(constant=-1.0, shift=1.0)
    precision = iter_precision['prec'] if 'prec' in iter_precision else PRECISION
    _run_sdc_with_problem(problem, sdc_core, num_time_steps, dt, num_nodes, iter_precision['iter'], precision)


def _lambda_u_function(sdc_core, num_time_steps, dt, num_nodes, iter_precision):
    problem = LambdaU(lmbda=complex(-1.0, 1.0))
    precision = iter_precision['prec'] if 'prec' in iter_precision else PRECISION
    _run_sdc_with_problem(problem, sdc_core, num_time_steps, dt, num_nodes, iter_precision['iter'], precision)


def test_constant_minus_one_function_with_explicit_sdc():
    _expected_iterations = {
        1: {
            1.0: {
                3: {'iter': 1},
                5: {'iter': 1},
                7: {'iter': 1}},
            0.5: {
                3: {'iter': 1},
                5: {'iter': 1},
                7: {'iter': 1}},
            (1.0/3.0): {
                3: {'iter': 1},
                5: {'iter': 1},
                7: {'iter': 1}
            }
        },
        2: {
            1.0: {
                3: {'iter': 1},
                5: {'iter': 1},
                7: {'iter': 1}
            }
        },
        3: {
            1.0: {
                3: {'iter': 1},
                5: {'iter': 1},
                7: {'iter': 1}
            }
        }
    }
    for _num_time_steps in _expected_iterations.keys():
        for _dt in _expected_iterations[_num_time_steps].keys():
            for _num_nodes in _expected_iterations[_num_time_steps][_dt].keys():
                yield _constant_minus_one_function, ExplicitSdcCore, \
                    _num_time_steps, _dt, _num_nodes, _expected_iterations[_num_time_steps][_dt][_num_nodes]


def test_constant_minus_one_function_with_implicit_sdc():
    _expected_iterations = {
        1: {
            1.0: {
                3: {'iter': 1},
                5: {'iter': 1},
                7: {'iter': 1}},
            0.5: {
                3: {'iter': 1},
                5: {'iter': 1},
                7: {'iter': 1}},
            (1.0/3.0): {
                3: {'iter': 1},
                5: {'iter': 1},
                7: {'iter': 1}
            }
        },
        2: {
            1.0: {
                3: {'iter': 1},
                5: {'iter': 1},
                7: {'iter': 1}
            }
        },
        3: {
            1.0: {
                3: {'iter': 1},
                5: {'iter': 1},
                7: {'iter': 1}
            }
        }
    }
    for _num_time_steps in _expected_iterations.keys():
        for _dt in _expected_iterations[_num_time_steps].keys():
            for _num_nodes in _expected_iterations[_num_time_steps][_dt].keys():
                yield _constant_minus_one_function, ImplicitSdcCore, \
                    _num_time_steps, _dt, _num_nodes, _expected_iterations[_num_time_steps][_dt][_num_nodes]


def test_constant_minus_one_function_with_semi_implicit_sdc():
    _expected_iterations = {
        1: {
            1.0: {
                3: {'iter': 1},
                5: {'iter': 1},
                7: {'iter': 1}},
            0.5: {
                3: {'iter': 1},
                5: {'iter': 1},
                7: {'iter': 1}},
            (1.0/3.0): {
                3: {'iter': 1},
                5: {'iter': 1},
                7: {'iter': 1}
            }
        },
        2: {
            1.0: {
                3: {'iter': 1},
                5: {'iter': 1},
                7: {'iter': 1}
            }
        },
        3: {
            1.0: {
                3: {'iter': 1},
                5: {'iter': 1},
                7: {'iter': 1}
            }
        }
    }
    for _num_time_steps in _expected_iterations.keys():
        for _dt in _expected_iterations[_num_time_steps].keys():
            for _num_nodes in _expected_iterations[_num_time_steps][_dt].keys():
                yield _constant_minus_one_function, SemiImplicitSdcCore, \
                    _num_time_steps, _dt, _num_nodes, _expected_iterations[_num_time_steps][_dt][_num_nodes]


def test_lambda_u_with_explicit_sdc():
    _expected_iterations = {
        1: {
            1.0: {
                3: {'iter': 22},
                5: {'iter': 12},
                7: {'iter': 9}},
            0.5: {
                3: {'iter': 19},
                5: {'iter': 7},
                7: {'iter': 6}},
            (1.0/3.0): {
                3: {'iter': 14},
                5: {'iter': 6},
                7: {'iter': 5}
            }
        },
        2: {
            1.0: {
                3: {'iter': 12},
                5: {'iter': 8},
                7: {'iter': 7}
            }
        },
        3: {
            1.0: {
                3: {'iter': 10},
                5: {'iter': 7},
                7: {'iter': 6}
            }
        }
    }
    for _num_time_steps in _expected_iterations.keys():
        for _dt in _expected_iterations[_num_time_steps].keys():
            for _num_nodes in _expected_iterations[_num_time_steps][_dt].keys():
                yield _lambda_u_function, ExplicitSdcCore, \
                    _num_time_steps, _dt, _num_nodes, _expected_iterations[_num_time_steps][_dt][_num_nodes]


def test_lambda_u_with_implicit_sdc():
    _expected_iterations = {
        1: {
            1.0: {
                3: {'iter': 12},
                5: {'iter': 10},
                7: {'iter': 8}},
            0.5: {
                3: {'iter': 16},
                5: {'iter': 7},
                7: {'iter': 6}},
            (1.0/3.0): {
                3: {'iter': 14},
                5: {'iter': 6},
                7: {'iter': 5}
            }
        },
        2: {
            1.0: {
                3: {'iter': 10},
                5: {'iter': 7},
                7: {'iter': 7}
            }
        },
        3: {
            1.0: {
                3: {'iter': 9},
                5: {'iter': 7},
                7: {'iter': 6}
            }
        }
    }
    for _num_time_steps in _expected_iterations.keys():
        for _dt in _expected_iterations[_num_time_steps].keys():
            for _num_nodes in _expected_iterations[_num_time_steps][_dt].keys():
                yield _lambda_u_function, ImplicitSdcCore, \
                    _num_time_steps, _dt, _num_nodes, _expected_iterations[_num_time_steps][_dt][_num_nodes]


def test_lambda_u_with_semi_implicit_sdc():
    _expected_iterations = {
        1: {
            1.0: {
                3: {'iter': 14},
                5: {'iter': 11},
                7: {'iter': 9}},
            0.5: {
                3: {'iter': 17},
                5: {'iter': 7},
                7: {'iter': 6}},
            (1.0/3.0): {
                3: {'iter': 15},
                5: {'iter': 6},
                7: {'iter': 6}
            }
        },
        2: {
            1.0: {
                3: {'iter': 10},
                5: {'iter': 8},
                7: {'iter': 7}
            }
        },
        3: {
            1.0: {
                3: {'iter': 9},
                5: {'iter': 7},
                7: {'iter': 6}
            }
        }
    }
    for _num_time_steps in _expected_iterations.keys():
        for _dt in _expected_iterations[_num_time_steps].keys():
            for _num_nodes in _expected_iterations[_num_time_steps][_dt].keys():
                yield _lambda_u_function, SemiImplicitSdcCore, \
                    _num_time_steps, _dt, _num_nodes, _expected_iterations[_num_time_steps][_dt][_num_nodes]


class SdcTest(NumpyAwareTestCase):
    def setUp(self):
        # self._test_obj = ParallelSdc()
        pass

    def test_semi_implicit_with_multi_dimensional(self):
        problem = Constant(constant=-1.0, shift=1.0, dim=(2, 3, 1))
        _run_sdc_with_problem(problem, SemiImplicitSdcCore, 1, 1.0, 3, 2, PRECISION)


if __name__ == "__main__":
    import unittest
    unittest.main()
