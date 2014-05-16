# coding=utf-8

from collections import OrderedDict

from pypint.solvers.parallel_sdc import ParallelSdc
from pypint.communicators.forward_sending_messaging import ForwardSendingMessaging
from pypint.integrators.sdc_integrator import SdcIntegrator
from pypint.problems.i_initial_value_problem import IInitialValueProblem
from pypint.utilities import assert_is_instance, assert_condition
from pypint.utilities.logging import *


def v_cycle_solver_factory(mg_problem, num_level):
    pass
