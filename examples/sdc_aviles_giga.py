# coding=utf-8

from examples.problems.aviles_giga import AvilesGiga
from pypint.solvers.cores import SemiImplicitSdcCore
from pypint.utilities.sdc_solver_factory import sdc_solver_factory
from pypint.utilities.threshold_check import ThresholdCheck
import matplotlib.pyplot as plt
n = 32
m = 2*n+1
problem = AvilesGiga(epsilon=0.1, n=n, time_start=0.0, time_end=1.0e-1)

def solve_with_factory(prob, core, num_solvers, num_total_time_steps):
    thresh = ThresholdCheck(max_threshold=20,
                            min_threshold=1e-7,
                            conditions=('residual', 'iterations'))
    solvers = sdc_solver_factory(prob, num_solvers, num_total_time_steps, core, threshold=thresh, num_time_steps=1, num_nodes=5)
    return solvers

solvers = solve_with_factory(problem, SemiImplicitSdcCore, 1, 20)
shape = (n*2+1, n*2+1)
solver = solvers[-1]
first = solver._states[0].initial.value
last = solver._states[-1].last.final_step.solution.value
fig = plt.figure(figsize=(16, 9))
for i in range(18):
    plt.subplot(3, 6, i+1)
    plt.yticks([])
    plt.xticks([])
    img = (solver._states[i].last.final_step.value).reshape(shape)
    plt.imshow(img)

# plt.tight_layout()
fig = plt.gcf()

fig.savefig("{:s}.png".format("aviles_giga"))
