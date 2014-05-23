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
    AvilesGiga(epsilon=0.1, n=16, time_start=0.0, time_end=1e-4)
]

def solve_with_factory(prob, core, num_solvers, num_total_time_steps):
    thresh = ThresholdCheck(max_threshold=5,
                            min_threshold=1e-7,
                            conditions=('residual', 'iterations'))
    solvers = sdc_solver_factory(prob, num_solvers, num_total_time_steps, core, threshold=thresh, num_time_steps=1, num_nodes=3)

for prob in problems:
    for core in [SemiImplicitSdcCore]:
        solve_with_factory(prob, core, 1, 10)
        print("RHS Evaluations: %d" % prob.rhs_evaluations)
        del prob.rhs_evaluations
        # solve_parallel(prob, core)
        # solve_parallel_two(prob, core)

solver=solvers[-1]
first=solvers[0].state.first.initial.value
plt.imshow(first.reshape(shape))
plt.colorbar(cmap=plt.cm.coolwarm)
plt.show()
