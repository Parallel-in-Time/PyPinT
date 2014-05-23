
# coding: utf-8

# In[92]:

from examples.problems.aviles_giga import AvilesGiga
from pypint.solvers.cores import SemiImplicitSdcCore
from pypint.utilities.sdc_solver_factory import sdc_solver_factory
from pypint.utilities.threshold_check import ThresholdCheck

n = 32
m = 2*n+1
problem = AvilesGiga(epsilon=0.1, n=n, time_start=0.0, time_end=1e-2)

def solve_with_factory(prob, core, num_solvers, num_total_time_steps):
    thresh = ThresholdCheck(max_threshold=15,
                            min_threshold=1e-7,
                            conditions=('residual', 'iterations'))
    solvers = sdc_solver_factory(prob, num_solvers, num_total_time_steps, core, threshold=thresh, num_time_steps=1, num_nodes=3)
    return solvers

solvers = solve_with_factory(problem, SemiImplicitSdcCore, 1, 5)
print("RHS Evaluations: %d" % problem.rhs_evaluations)


# In[93]:

solver=solvers[-1]


# In[94]:

last=solver._states[-1].last.final_step.solution.value


# In[95]:

first=solvers[0]._states[0].initial.value



import matplotlib.pyplot as plt


# In[99]:

shape = (m, m)
plt.imshow(last.reshape(shape))
plt.colorbar(cmap=plt.cm.coolwarm)
plt.show()


# In[100]:

plt.imshow(first.reshape(shape))
plt.colorbar(cmap=plt.cm.coolwarm)
plt.show()


# In[101]:

plt.imshow(first.reshape(shape) - last.reshape(shape))
plt.colorbar(cmap=plt.cm.coolwarm)
plt.show()


# In[101]:



