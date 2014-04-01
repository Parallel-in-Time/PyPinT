# coding=utf-8
"""
.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
from collections import OrderedDict

from pypint.solvers.parallel_sdc import ParallelSdc
from pypint.communicators.forward_sending_messaging import ForwardSendingMessaging
from pypint.integrators.sdc_integrator import SdcIntegrator
from pypint.problems.i_initial_value_problem import IInitialValueProblem
from pypint.utilities import assert_is_instance, assert_condition
from pypint.utilities.logging import *


def sdc_solver_factory(problem, num_solvers, num_total_time_steps, solver_core, **solver_options):
    """Factory function for Parallel SDC with Forward Sending Messaging

    This function creates, initializes and executes one or more SDC solvers in parallel for a given problem.
    The number of solver instances and total number of time steps can be specified as well as the type of SDC core.

    Parameters
    ----------
    problem : :py:class:`.IInitialValueProblem`
        the initial value problem to create the solvers for
    num_solvers : :py:class:`int`
        number of parallel solvers to be created
    num_total_time_steps : :py:class:`int`
        total number of time steps for the whole interval defined by the problem
    solver_core : :py:class:`.SdcSolverCore`
        type of the SDC solver core
    solver_options : :py:class:`dict`
        options to be passed as it to the solver instantiation
        (see :py:meth:`.ParallelSDC.__init__` for details)

    Returns
    -------
    solvers : :py:class:`list` of :py:class:`.ParallelSDC`

    Raises
    ------
    ValueError

        * if ``problem`` is not an :py:class:`.IInitialValueProblem`
        * if ``num_solvers`` is not an :py:class:`int` or not larger zero
        * if ``num_total_time_steps`` is smaller than ``num_solvers``
        * if ``solver_options`` is not a :py:class:`dict`
        * if the interval width per solver core is invalid (i.e. not non-zero possitive or larger the problem width)
    """
    assert_is_instance(problem, IInitialValueProblem, descriptor="Problem")
    assert_is_instance(num_solvers, int, descriptor="Number of Desired Solvers")
    assert_condition(num_solvers > 0,
                     ValueError, message="Number of solvers must be greater 0: NOT %d" % num_solvers)
    assert_is_instance(num_total_time_steps, int, descriptor="Total Number of Time Steps")
    assert_condition(num_total_time_steps >= num_solvers,
                     ValueError, message=("Total Number of Time Steps must be at least as large as number of solvers: "
                                          "%d < %d" % (num_total_time_steps, num_solvers)))
    assert_is_instance(solver_options, dict, descriptor="Solver Options")

    _log_messages = OrderedDict({'': OrderedDict()})

    LOG.info("%s%s" % (VERBOSITY_LVL1, SEPARATOR_LVL1))
    LOG.info("{}{:#<80}".format(VERBOSITY_LVL1, "START: %s " % solver_core.name))

    _log_messages['']['Problem'] = problem.print_lines_for_log()

    # calculate width of interval per solver call
    _prob_width = (problem.time_end - problem.time_start)
    if 'num_time_steps' in solver_options:
        # multiple time steps per solver
        _total_num_calls = num_total_time_steps / solver_options['num_time_steps']
    else:
        # one time step per solver
        _total_num_calls = num_total_time_steps
    LOG.debug("Total number of solver calls: %d" % _total_num_calls)

    _dt = _prob_width / float(_total_num_calls)
    assert_condition(0.0 < _dt <= _prob_width,
                     RuntimeError,
                     message="Width of interval per solver is invalid: 0.0 > %f <= %f" % (_dt, _prob_width))
    assert_condition(abs((_dt * _total_num_calls) - _prob_width) <= 1e-16,
                     ValueError,
                     message="Number of solver calls and calculated interval width do not make sense.")
    LOG.debug("Interval width per solver call: %f" % _dt)

    _log_messages['']['Number Solver Instances'] = "%d" % num_solvers
    _log_messages['']['Interval Width per Solver Call'] = "{:.3f}".format(_dt)
    _log_messages['']['Total Number Solver Calls'] = "%d" % _total_num_calls

    # list of solvers and their communicators
    # (list indices associate solvers and communicators)
    _solvers = []
    _comms = []

    # instantiate communicators and solvers
    for _n in range(0, num_solvers):
        _comms.append(ForwardSendingMessaging())
        _solvers.append(ParallelSdc(communicator=_comms[-1]))

    # write problem's initial values into the first communicator
    _comms[0].write_buffer(value=problem.initial_value, time_point=problem.time_start)

    # link communicators
    if num_solvers > 1:
        for _n in range(0, num_solvers):
            if _n != num_solvers - 1:
                # all but the very last
                # (index arithmetics for first communicator is included)
                _comms[_n].link_solvers(previous=_comms[_n - 1], next=_comms[_n + 1])
            else:
                # last communicator
                _comms[_n].link_solvers(previous=_comms[_n - 1], next=_comms[0])
    else:
        # we need a special handling in case of only one solver/communicator
        _comms[0].link_solvers(previous=_comms[0], next=_comms[0])

    # initialize solvers
    for _s in _solvers:
        _s.init(problem=problem, integrator=SdcIntegrator, **solver_options)

    _log_messages['']['Individual Solver'] = _solvers[0].print_lines_for_log()

    print_logging_message_tree(_log_messages)

    # run solvers
    _calls = []
    while len(_calls) < _total_num_calls:
        for _s in _solvers:
            if len(_calls) == _total_num_calls:
                LOG.debug("Problem done.")
                break
            LOG.info("%s%s" % (VERBOSITY_LVL1, SEPARATOR_LVL1))
            LOG.info("%sCalling solver %d" % (VERBOSITY_LVL1, _solvers.index(_s)))
            _s.run(core=solver_core, dt=_dt)
            _calls.append(_solvers.index(_s))
    LOG.info("%s%s" % (VERBOSITY_LVL1, SEPARATOR_LVL1))
    LOG.info("%sLast solver called: %d" % (VERBOSITY_LVL2, _calls[-1]))

    return _solvers


__all__ = ['sdc_solver_factory']
