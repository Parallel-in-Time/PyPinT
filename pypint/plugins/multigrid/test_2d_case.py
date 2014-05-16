# coding=utf-8

# using the MultiGridLevel2D class we

import numpy as np
import sys
print(sys.path)
from pypint.plugins.multigrid.multigrid_problem import MultiGridProblem
from pypint.plugins.multigrid.multigrid_level_provider import MultiGridLevelProvider
from pypint.plugins.multigrid.multigrid_solution import MultiGridSolution
from pypint.plugins.multigrid.level2d import MultigridLevel2D
from pypint.plugins.multigrid.multigrid_smoother import ILUSmoother, SplitSmoother, DirectSolverSmoother, WeightedJacobiSmoother
from pypint.utilities import assert_is_callable, assert_is_instance, assert_condition
from pypint.plugins.multigrid.stencil import Stencil
from pypint.plugins.multigrid.interpolation import InterpolationByStencilListIn1D, InterpolationByStencilForLevels, InterpolationByStencilForLevelsClassical
from pypint.plugins.multigrid.restriction import RestrictionStencilPure, RestrictionByStencilForLevels, RestrictionByStencilForLevelsClassical
from operator import iadd,add
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

# class SmootherControl(object):
#     """
#
#     """
#     def __init__(self, level_list, SmootherTypeList=["wJac"], ):
#
#



class VCycler(object):
    """ Takes a problem and constructs a level set a stencil set and so on to do v cycles

    """
    def __init__(self, stencil_form, mg_problem, numb_levels=3, coarse_level_shape=(5,)):
        pass



if __name__ == '__main__':
    # check if level2d is working properly
    # but first one has to define a proper level
    # and for this one needs a useful stencil

    np.set_printoptions(precision=4, edgeitems=4, threshold=10)

    print("===== Stencil =====")
    laplace_array = np.asarray([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
    laplace_stencil = Stencil(np.asarray([[0, 1, 0], [1, -4, 1], [0, 1, 0]]), None, 2)
    print("stencil.arr\n", laplace_stencil.arr)
    print("stencil.reversed_arr\n", laplace_stencil.reversed_arr)
    print("stencil.b \n", laplace_stencil.b)
    print("stencil.positions \n", laplace_stencil.positions)
    print("stencil.center \n", laplace_stencil.center)
    print("===== MgProblem =====")
    # define geometry
    geo = np.asarray([[0, 1], [0, 1]])
    # the boundary conditions, in this case dirichlet boundary conditions
    boundary_type = ["dirichlet"]*2
    # east_f = lambda x: 2.0
    # west_f = lambda x: 8.0
    # north_f = lambda x: 1.0
    # south_f = lambda x: 4.0
    east_f = lambda x: np.sin(x[1]*np.pi)
    west_f = lambda x: np.sin(x[1]*np.pi)
    north_f = lambda x: np.sin(x[0]*np.pi)
    south_f = lambda x: np.sin(x[0]*np.pi)
    boundary_functions = [[west_f, east_f], [north_f, south_f]]
    rhs_function = lambda x, y: 0.0

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
    print("level.north_tensor \n", level.north_tensor)
    print("level.west_tensor \n", level.west_tensor)
    print("level.east_tensor \n", level.east_tensor)
    print("level.se_tensor \n", level.se_tensor)
    print("level.ne_tensor \n", level.ne_tensor)
    print("level.sw_tensor \n", level.sw_tensor)
    print("level.nw_tensor \n", level.nw_tensor)

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

    direct_solver = DirectSolverSmoother(laplace_stencil, level)
    laplace_stencil.modify_rhs(level)
    direct_solver.relax()
    print("level.arr after direct solution\n", level.mid)
    print("test of the solution Ax=b by convolve\n ", laplace_stencil.eval_convolve(level.mid, "same"))
    rhs_test = np.zeros(level.rhs.shape)
    laplace_stencil.eval_sparse(level.mid, rhs_test)
    print("test of the solution Ax=b by sparse matrix application \n", rhs_test)

    print("==== SplitSmoother ====")
    omega = 2.0/3.0
    l_plus = np.asarray([[0, 0, 0],
                         [0, -4.0/omega, 0],
                         [0, 0, 0]])
    l_minus = np.asarray([[0, 1.0, 0], [1.0, -4.0*(1.0 - 1.0/omega), 1.0], [0., 1., 0.]])

    jacobi_smoother = SplitSmoother(l_plus, l_minus, level)
    level.mid[:] = 0
    jacobi_smoother.relax()
    print("level.arr after one jacobi step with modified rhs\n", level.arr)

    level.mid[:] = 0.
    level.modified_rhs = False
    level.rhs[:] = 0.
    jacobi_smoother = SplitSmoother(l_plus, l_minus, level)
    jacobi_smoother.relax()
    print("level.arr after one jacobi step with unmodified rhs\n", level.arr)

    # For the test of the level transitioning we define 3 levels
    # with different roles but the same borders
    n_jacobi_pre = 5
    n_jacobi_post = 5
    borders = np.ones((2, 2))
    top_level = MultigridLevel2D((259, 259), mg_problem=mg_problem,
                                 max_borders=borders, role="FL")



    mid_level = MultigridLevel2D((129, 129), mg_problem=mg_problem,
                                 max_borders=borders, role="ML")

    low_level = MultigridLevel2D((64, 64), mg_problem=mg_problem,
                                 max_borders=borders, role="CL")
    mg_problem.fill_rhs(top_level)
    top_level.pad()


    # define the different stencils
    top_stencil = Stencil(laplace_array/top_level.h[0]**2, None, 2)
    mid_stencil = Stencil(laplace_array/mid_level.h[0]**2, None, 2)
    low_stencil = Stencil(laplace_array/low_level.h[0]**2, None, 2)
    top_stencil.modify_rhs(top_level)

    # define the different smoothers on each level, works just for symmetric grids
    top_jacobi_smoother = SplitSmoother(l_plus / top_level.h[0]**2,
                                        l_minus / top_level.h[0]**2,
                                        top_level)
    mid_jacobi_smoother = SplitSmoother(l_plus / mid_level.h[0]**2,
                                        l_minus / mid_level.h[0]**2,
                                        mid_level)
    low_jacobi_smoother = SplitSmoother(l_plus / low_level.h[0]**2,
                                        l_minus / low_level.h[0]**2,
                                        low_level)
    low_direct_smoother = DirectSolverSmoother(low_stencil, low_level)

    # prepare the ilu smoother
    top_ilu_smoother = ILUSmoother(top_stencil, top_level)
    mid_ilu_smoother = ILUSmoother(mid_stencil, mid_level)
    low_ilu_smoother = ILUSmoother(low_stencil, low_level)

    # define the the restriction operators - full weighting
    rst_stencil = Stencil(np.asarray([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]])/16)
    rst_top_to_mid = RestrictionByStencilForLevelsClassical(top_level, mid_level, rst_stencil)
    rst_mid_to_low = RestrictionByStencilForLevelsClassical(mid_level, low_level, rst_stencil)

    # define the interpolation
    corner_array = np.ones((2., 2.)) * 0.25
    border_arr_h = np.asarray([[0.5, 0.5]])
    border_arr_v = np.asarray([[0.5], [0.5]])

    ipl_stencil_list = [(Stencil(np.asarray([[1]])), (1, 1)),
                        (Stencil(corner_array), (0, 0)),
                        (Stencil(border_arr_h), (1, 0)),
                        (Stencil(border_arr_v), (0, 1))]

    ipl_mid_to_top = InterpolationByStencilForLevelsClassical(mid_level, top_level, ipl_stencil_list, pre_assign=iadd)
    ipl_low_to_mid = InterpolationByStencilForLevelsClassical(low_level, mid_level, ipl_stencil_list, pre_assign=iadd)

    print("==== Down The V Cycle ====")
    print("** Initial TopLevel.arr **\n", top_level.arr)
    # top_jacobi_smoother.relax(n_jacobi_pre)
    # print("** TopLevel.arr after "+str(n_jacobi_pre)+" jacobi step(s) **\n", top_level.arr)

    top_ilu_smoother.relax(n_jacobi_pre)
    print("** TopLevel.arr after "+str(n_jacobi_pre)+" ilu step(s) **\n", top_level.arr)

    print("** TopLevel.res before computation **\n", top_level.res)
    top_level.compute_residual(top_stencil)
    print("** TopLevel.res after computation **\n", top_level.res)
    print("** MidLevel.rhs before restriction **\n", mid_level.rhs)
    rst_top_to_mid.restrict()
    print("** MidLevel.rhs after restriction **\n", mid_level.rhs)

    # mid_jacobi_smoother.relax(n_jacobi_pre)
    # print("** MidLevel.arr after "+str(n_jacobi_pre)+" jacobi step(s) **\n", mid_level.arr)

    mid_ilu_smoother.relax(n_jacobi_pre)
    print("** MidLevel.arr after "+str(n_jacobi_pre)+" ilu step(s) **\n", mid_level.arr)

    print("** MidLevel.res before computation **\n", mid_level.res)
    mid_level.compute_residual(mid_stencil)
    print("** MidLevel.res after computation **\n", mid_level.res)

    print("** LowLevel.rhs before restriction **\n", low_level.rhs)
    rst_mid_to_low.restrict()
    print("** LowLevel.rhs after restriction **\n", low_level.rhs)
    # low_jacobi_smoother.relax(n_jacobi_pre)
    # print("** LowLevel.arr after "+str(n_jacobi_pre)+" jacobi step(s) **\n", low_level.arr)
    low_direct_smoother.relax()
    # low_level.mid[:] = (np.arange(4).reshape((2, 2)) + 1)**2
    print("** LowLevel.arr after direct solve **\n", low_level.arr)
    print("** MidLevel.arr before interpolation **\n", mid_level.arr)
    # mid_level.mid[:] = 0.0
    ipl_low_to_mid.eval()
    print("** MidLevel.arr after interpolation **\n", mid_level.arr)
    # mid_jacobi_smoother.relax(n_jacobi_post)
    # print("** MidLevel.arr after "+str(n_jacobi_post)+" jacobi step(s) **\n", mid_level.arr)

    mid_ilu_smoother.relax(n_jacobi_post)
    print("** MidLevel.arr after "+str(n_jacobi_post)+" ilu step(s) **\n", mid_level.arr)

    ipl_mid_to_top.eval()
    print("** TopLevel.arr after interpolation **\n", top_level.arr)

    # top_jacobi_smoother.relax(n_jacobi_post)
    # print("** TopLevel.arr after "+str(n_jacobi_post)+" jacobi step(s) **\n", top_level.arr)

    top_ilu_smoother.relax(n_jacobi_post)
    print("** TopLevel.arr after "+str(n_jacobi_post)+" ilu step(s) **\n", top_level.arr)

    sol_level = MultigridLevel2D((259, 259),
                             mg_problem=mg_problem,
                             max_borders=np.asarray([[1, 1], [1, 1]]),
                             role="FL")
    sol_level.rhs[:] = 0.0
    sol_level.pad()
    sol_stencil = Stencil(laplace_array/sol_level.h[0]**2, None, 2)
    sol_stencil.modify_rhs(sol_level)
    # print("I am modified", sol_level.modified_rhs)
    direct_solver = DirectSolverSmoother(sol_stencil, sol_level)
    direct_solver.relax()
    print("** The collocation solution **\n", sol_level.arr)
    print("** The error **\n", np.max(np.abs(sol_level.mid - top_level.mid)))
    print()
    print("maximum value\n", np.max(top_level.mid))
    print(np.max(sol_level.mid))
    # plot the data
    # shade data, creating an rgb array.
    # plot un-shaded and shaded images.
    plt.imshow(top_level.mid, cmap=plt.cm.coolwarm)
    plt.title('Sinus Randbedingungen')
    plt.xticks([]); plt.yticks([])
    plt.colorbar(cmap=plt.cm.coolwarm)
    plt.show()
