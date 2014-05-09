# coding=utf-8

# using the MultiGridLevel2D class we

import numpy as np
import sys
print(sys.path)
from pypint.plugins.multigrid.multigrid_problem import MultiGridProblem
from pypint.plugins.multigrid.multigrid_level_provider import MultiGridLevelProvider
from pypint.plugins.multigrid.multigrid_solution import MultiGridSolution
from pypint.plugins.multigrid.level2d import MultigridLevel2D
from pypint.plugins.multigrid.multigrid_smoother import SplitSmoother, DirectSolverSmoother, WeightedJacobiSmoother
from pypint.utilities import assert_is_callable, assert_is_instance, assert_condition
from pypint.plugins.multigrid.stencil import Stencil
from pypint.plugins.multigrid.interpolation import InterpolationByStencilListIn1D, InterpolationByStencilForLevels, InterpolationByStencilForLevelsClassical
from pypint.plugins.multigrid.restriction import RestrictionStencilPure, RestrictionByStencilForLevels, RestrictionByStencilForLevelsClassical
from operator import iadd,add

if __name__ == '__main__':
    # check if level2d is working properly
    # but first one has to define a proper level
    # and for this one needs a useful stencil


    print("===== Stencil =====")
    laplace_stencil = Stencil(np.asarray([[0, 1, 0], [1, -4, 1], [0, 1, 0]]), None, 2)
    print("stencil.arr\n", laplace_stencil.arr)
    print("stencil.reversed_arr\n", laplace_stencil.reversed_arr)
    print("stencil.b \n", laplace_stencil.b)
    print("stencil.positions \n", laplace_stencil.positions)

    print("===== MgProblem =====")
    # define geometry
    geo = np.asarray([[0, 1], [0, 1]])
    # the boundary conditions, in this case dirichlet boundary conditions
    boundary_type = ["dirichlet"]*2
    east_f = lambda x: 2.0
    west_f = lambda x: 8.0
    north_f = lambda x: 1.0
    south_f = lambda x: 4.0
    boundary_functions = [[west_f, east_f], [north_f, south_f]]
    rhs_function = lambda x: 0.0

    mg_problem = MultiGridProblem(laplace_stencil,
                                  rhs_function,
                                  boundary_functions=boundary_functions,
                                  boundaries="dirichlet",
                                  geometry=geo)

    print("Constructed SpaceTensor\n", mg_problem.construct_space_tensor(10))
    print("mg_problem.geometry", mg_problem.geometry)
    print("mg_problem.boundaries", mg_problem.boundaries)
    print("===== MultiGridLevel2d =====")
    level = MultigridLevel2D((4, 4),
                             mg_problem=mg_problem,
                             max_borders=np.asarray([[1, 1], [1, 1]]),
                             role="FL")

    print("level.arr \n", level.arr)
    print("level.mid \n", level.mid)
    print("level.south \n", level.south)
    print("level.se \n", level.se)

    print("level.h\n", level.h)

    print("level.space_tensor \n", level.space_tensor)
    print("level.mid_tensor \n", level.mid_tensor)
    print("level.south_tensor \n", level.south_tensor)
    print("level.se_tensor \n", level.se_tensor)

    level.pad()
    print("level.arr after padding\n", level.arr)
    print("north_east\n", level.ne)
    print("north_west\n", level.nw)
    print("south_east\n", level.se)
    print("south_west\n", level.sw)

    print("evaluable view of stencil\n", level.evaluable_view(laplace_stencil))
    is_on_border = level.border_function_generator(laplace_stencil)
    border_truth = [[is_on_border((i, j)) for j in range(level.arr.shape[1])] for i in range(level.arr.shape[0])]
    print("border_truth\n", np.asarray(border_truth, dtype=np.int))

    level.rhs[:] = 0.0
    laplace_stencil.modify_rhs(level)
    print("level.rhs after modification\n", level.rhs)

    print("==== DirectSolver ====")
    omega = 0.5
    l_plus = np.asarray([0, -2.0/omega, 0])
    l_minus = np.asarray([1.0, -2.0*(1.0 - 1.0/omega), 1.0])
    direct_solver = DirectSolverSmoother(laplace_stencil, level)
    laplace_stencil.modify_rhs(level)
    direct_solver.relax()
    print("level.arr after direct solution\n", level.mid)
    print("test of the solution Ax=b by convolve\n ", laplace_stencil.eval_convolve(level.mid, "same"))
    rhs_test = np.zeros(level.rhs.shape)
    laplace_stencil.eval_sparse(level.mid, rhs_test)
    print("test of the solution Ax=b by sparse matrix application \n", rhs_test)

    print("==== SplitSmoother ====")
