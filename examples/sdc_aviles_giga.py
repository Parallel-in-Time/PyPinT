# coding=utf-8
import numpy as np

from examples.problems.constant import Constant
from examples.problems.lambda_u import LambdaU
from examples.problems.aviles_giga import AvilesGiga
from pypint.communicators.forward_sending_messaging import ForwardSendingMessaging
from pypint.solvers.parallel_sdc import ParallelSdc
from pypint.integrators.sdc_integrator import SdcIntegrator
from pypint.solvers.cores import ExplicitSdcCore, ImplicitSdcCore, SemiImplicitSdcCore
from pypint.utilities.sdc_solver_factory import sdc_solver_factory
from pypint.utilities.threshold_check import ThresholdCheck

problems = [
    # Constant(constant=-1.0),
    AvilesGiga(epsilon=0.3, n=)
]

def solve_with_factory(prob, core, num_solvers, num_total_time_steps):
    thresh = ThresholdCheck(max_threshold=25,
                            min_threshold=1e-7,
                            conditions=('error', 'error reduction', 'residual', 'iterations'))
    solvers = sdc_solver_factory(prob, num_solvers, num_total_time_steps, core, threshold=thresh, num_time_steps=2, num_nodes=3)

def solve_parallel(prob, _core):
    thresh = ThresholdCheck(max_threshold=25,
                            min_threshold=1e-7,
                            conditions=('error', 'solution reduction', 'error reduction', 'residual', 'iterations'))
    comm = ForwardSendingMessaging()
    solver = ParallelSdc(communicator=comm)
    comm.link_solvers(previous=comm, next=comm)
    comm.write_buffer(value=prob.initial_value, time_point=prob.time_start)
    solver.init(integrator=SdcIntegrator, threshold=thresh, problem=prob, num_time_steps=1, num_nodes=3)
    sol = solver.run(core=_core, dt=0.5)
    print(sol)

def solve_parallel_two(prob, _core):
    thresh = ThresholdCheck(max_threshold=25,
                            min_threshold=1e-7,
                            conditions=('error', 'solution reduction', 'error reduction', 'residual', 'iterations'))
    comm1 = ForwardSendingMessaging()
    comm2 = ForwardSendingMessaging()
    solver1 = ParallelSdc(communicator=comm1)
    solver2 = ParallelSdc(communicator=comm2)
    comm1.link_solvers(previous=comm2, next=comm2)
    comm1.write_buffer(value=prob.initial_value, time_point=prob.time_start)
    comm2.link_solvers(previous=comm1, next=comm1)
    solver1.init(integrator=SdcIntegrator, threshold=thresh, problem=prob, num_time_steps=1, num_nodes=3)
    solver2.init(integrator=SdcIntegrator, threshold=thresh, problem=prob, num_time_steps=1, num_nodes=3)

    solver1.run(core=_core, dt=0.25)
    solver2.run(core=_core, dt=0.25)
    solver1.run(core=_core, dt=0.25)
    solver2.run(core=_core, dt=0.25)

for prob in problems:
    for core in [SemiImplicitSdcCore]:
        solve_with_factory(prob, core, 1, 1)
        print("RHS Evaluations: %d" % prob.rhs_evaluations)
        del prob.rhs_evaluations
        # solve_parallel(prob, core)
        # solve_parallel_two(prob, core)
