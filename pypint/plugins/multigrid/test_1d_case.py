# coding=utf-8
import numpy as np
import sys
print(sys.path)
from pypint.plugins.multigrid.multigrid_problem import MultigridProblem
from pypint.plugins.multigrid.multigrid_level_provider import MultiGridLevelProvider
from pypint.plugins.multigrid.multigrid_solution import MultiGridSolution
from pypint.plugins.multigrid.level import MultigridLevel1D
from pypint.plugins.multigrid.multigrid_smoother import SplitSmoother, DirectSolverSmoother, WeightedJacobiSmoother
from pypint.utilities import assert_is_callable, assert_is_instance, assert_condition
from pypint.plugins.multigrid.stencil import Stencil
from pypint.plugins.multigrid.interpolation import InterpolationByStencilListIn1D, InterpolationByStencilForLevels, InterpolationByStencilForLevelsClassical
from pypint.plugins.multigrid.restriction import RestrictionStencilPure, RestrictionByStencilForLevels, RestrictionByStencilForLevelsClassical
from operator import iadd,add
import networkx as nx

if __name__ == '__main__':
    print("Lets solve, you guessed it, the heat equation")
    # heat equation needs a stencil
    laplace_stencil = Stencil(np.asarray([1, -2, 1]), None, 2)
    # test stencil to some extend
    print("===== Stencil tests =====")
    print("stencil.b :", laplace_stencil.b)
    print("stencil.positions :", laplace_stencil.positions)

    for i in laplace_stencil.positions:
        print(laplace_stencil.arr[i])

    print("stencil.relative_position :", laplace_stencil.relative_positions)
    print("stencil.relative_position_woc :", laplace_stencil.relative_positions_woc)
    # geometry is a 1 dimensional line
    geo = np.asarray([[0, 1]])
    print(geo.shape)
    # the boundary conditions, in this case dirichlet boundary conditions
    boundary_type = ["dirichlet"]*2
    left_f = lambda x: 100.0
    right_f = lambda x: 110.0
    boundary_functions = [[left_f, right_f]]

    def stupid_f(*args, **kwargs):
        return 0.0

    rhs_function = lambda x: 0.0
    mg_problem = MultigridProblem(laplace_stencil,
                                  stupid_f,
                                  boundary_functions=boundary_functions,
                                  boundaries=boundary_type,
                                  geometry=geo)
    # test some of the methods of mg_problem

    print("Mid of stencil method", mg_problem.mid_of_stencil(laplace_stencil))
    print("Eval_Convolve ([100,105,105,105,110])", laplace_stencil.eval_convolve(np.asarray([100, 105, 105, 105, 110])))

    print("===== MultigridProblemTest =====")
    print("Constructed SpaceTensor", mg_problem.construct_space_tensor(12))
    print("Checked if the grid distances are right",
          mg_problem.act_grid_distances)

    # they work properly at least for the 1d case
    # lets define the different levels lets try 3
    borders = np.asarray([3, 3])

    top_level = MultigridLevel1D(513, mg_problem=mg_problem,
                                 max_borders=borders, role="FL")

    mid_level = MultigridLevel1D(257, mg_problem=mg_problem,
                                 max_borders=borders, role="ML")

    low_level = MultigridLevel1D(65, mg_problem=mg_problem,
                                 max_borders=borders, role="CL")
    # check if the distance between points is calculated right
    print("===== MultigridLevel Test =====")
    print("3 different GridDistances from top to low level:")
    print(top_level.h)
    print(mid_level.h)
    print(low_level.h)
    print("the space_tensor of the last level")
    print(low_level.space_tensor)
    print(*mg_problem.act_grid_distances)
    print("test the is on border function")
    is_on_border = low_level.border_function_generator(laplace_stencil)
    border_truth = [is_on_border((i,)) for i in range(low_level.evaluable_view(laplace_stencil).size)]
    print(border_truth)

    # define the smoother from the split smoother class on each level,
    # where the last level is solved directly
    # omega = 1/np.sqrt(2)
    omega = 0.5
    l_plus = np.asarray([0, -2.0/omega, 0])
    l_minus = np.asarray([1.0, -2.0*(1.0 - 1.0/omega), 1.0])
    top_jacobi_smoother = SplitSmoother(l_plus / top_level.h**2,
                                        l_minus / top_level.h**2,
                                        top_level)
    mid_jacobi_smoother = SplitSmoother(l_plus / mid_level.h**2,
                                        l_minus / mid_level.h**2,
                                        mid_level)
    low_jacobi_smoother = SplitSmoother(l_plus / low_level.h**2,
                                        l_minus / low_level.h**2,
                                        low_level)
    low_direct_smoother = DirectSolverSmoother(laplace_stencil, low_level)
    # time to test the relaxation methods
    print("===== DirectSolverSmoother Test =====")
    low_level.rhs[:] = 0.0
    low_level.pad()
    print("arr:", low_level.arr)
    laplace_stencil.modify_rhs(low_level)
    print("rhs:", low_level.rhs)
    low_direct_smoother.relax()
    print(low_level.arr)
    low_level.pad()
    # Lets test the SplitSmoother by using the jacobi smoother
    # but for this case we need an initial guess
    print("===== JacobiSmoother Test =====")
    # define the 3 different JacobiSmoother Implementations
    jacobi_loop = WeightedJacobiSmoother(laplace_stencil,
                                         low_level, 0.5, "loop")
    jacobi_matrix = WeightedJacobiSmoother(laplace_stencil,
                                           low_level, 0.5, "matrix")
    jacobi_convolve = WeightedJacobiSmoother(laplace_stencil,
                                            low_level, 0.5, "convolve")

    low_level.mid[:] = 105.0
    low_level.pad()
    print("We start with this initial condition:")
    print(low_level.arr)
    print("Now we do a jacobi step using the convolve of stencil:")
    # funktioniert beinahe ist nur falsch rum
    jacobi_convolve.relax()
    print(low_level.arr)
    low_level.mid[:] = 105.0
    low_level.pad()
    # print("Wurde hier der stencil veraendert?", laplace_stencil.arr)
    print("Now we do a jacobi step using simple loops:")
    jacobi_loop.relax()
    print(low_level.arr)
    low_level.mid[:] = 105.0
    low_level.pad()

    print("Now we do a jacobi step using sparse matrix algorithms:")
    print("But just before one needs to fill the rhs")
    print("rhs at the beginning: \n", low_level.rhs)
    mg_problem.fill_rhs(low_level)
    print("rhs after filling it: \n", low_level.rhs)
    # laplace_stencil.modify_rhs(low_level)
    print("rhs after modification: \n", low_level.rhs)
    jacobi_matrix.relax()
    print(low_level.arr)

    print("Now we check if the more general SplitSmootherClass :")
    mg_problem.fill_rhs(low_level)
    low_level.mid[:] = 105.0
    low_level.pad()
    low_jacobi_smoother.relax()
    print(low_level.arr)

    print("===== Restriction and Interpolation =====")
    # generate restriction stencil
    rst_inject = RestrictionStencilPure(np.asarray([1.0]), 2)
    rst_fw = RestrictionStencilPure(np.asarray([0.25, 0.5, 0.25]), 2)
    # try it with just some simple arrays
    x_in = np.arange(9) ** 2
    x_out = np.zeros(5)
    print("test injection,\n x_in :", x_in)
    print(" x_out :", x_out)
    rst_inject.eval(x_in, x_out)
    print("inject,\n x_out :", x_out)
    # full weighting restriction needs another interpretation because
    # the stencil needs also the values on the boundary, this men
    x_out = np.zeros(4)
    print("full weighting needs another interpretation, so the x_out needs another size\n x_out", x_out)
    rst_fw.eval(x_in, x_out)
    print("After restriction,\n x_out:", x_out)

    # next we try the RestrictionStencilForLevels

    rst_lvl = RestrictionByStencilForLevels(Stencil(np.asarray([0.25, 0.5, 0.25])), mid_level, low_level)
    mid_level.restrict_out[:] = np.arange(mid_level.restrict_out.shape[0])
    print("mid level with borders before restriction : \n", mid_level.restrict_out)
    print("low level before restriction : \n", low_level.restrict_in)
    rst_lvl.restrict()
    print("low level after restriction : \n", low_level.restrict_in)
    print("it works horray, now the interpolation, first the 4 interpolation stencils which are nedded:")
    center = np.asarray([0])
    ipl_stencil_list = [(Stencil(np.asarray([1]), center), (0,)),
                        (Stencil(np.asarray([0.75, 0.25]), center), (1,)),
                        (Stencil(np.asarray([0.5, 0.5]), center), (2,)),
                        (Stencil(np.asarray([0.25, 0.75]), center), (3,))]
    print(ipl_stencil_list)
    mid_level.interpolate_in[:] = 0.0
    low_level.mid[:] = low_level.rhs[:]
    print("lowlevel before interpolation : \n", low_level.interpolate_out)
    print("midlevel before interpolation : \n", mid_level.interpolate_in)
    ipl_by_stencil = InterpolationByStencilForLevels(ipl_stencil_list, low_level, mid_level)
    ipl_by_stencil.eval()
    print("midlevel after interpolation : \n", mid_level.interpolate_in)

    print("========= Now we watch how the parts work together ==============")
    # because of testing reasons the nodes on each level are chosen different, so the classical interpolation and
    # restriction is possible
    top_level = MultigridLevel1D(67, mg_problem=mg_problem,
                                 max_borders=borders, role="FL")

    mid_level = MultigridLevel1D(33, mg_problem=mg_problem,
                                 max_borders=borders, role="ML")

    low_level = MultigridLevel1D(16, mg_problem=mg_problem,
                                 max_borders=borders, role="CL")
    # hence we need new smoothers

    top_jacobi_smoother = SplitSmoother(l_plus / top_level.h**2,
                                        l_minus / top_level.h**2,
                                        top_level, modified_rhs=True)
    mid_jacobi_smoother = SplitSmoother(l_plus / mid_level.h**2,
                                        l_minus / mid_level.h**2,
                                        mid_level, order=2)
    low_jacobi_smoother = SplitSmoother(l_plus / low_level.h**2,
                                        l_minus / low_level.h**2,
                                        low_level, order=2)
    low_direct_smoother = DirectSolverSmoother(laplace_stencil, low_level)
    top_direct_solver = DirectSolverSmoother(laplace_stencil, top_level)

    # first set initial values for the coarser levels to zero

    n_jacobi_pre = 1
    n_jacobi_post = 1
    # we define the Restriction operator
    rst_stencil = Stencil(np.asarray([0.25, 0.5, 0.25]))
    rst_top_to_mid = RestrictionByStencilForLevelsClassical(rst_stencil, top_level, mid_level)
    rst_mid_to_low = RestrictionByStencilForLevelsClassical(rst_stencil, mid_level, low_level)

    # and the interpolation operator
    ipl_stencil_list_standard = [(Stencil(np.asarray([1]), center), (1,)),
                                   (Stencil(np.asarray([0.5, 0.5]), center), (0,))]

    ipl_mid_to_top = InterpolationByStencilForLevelsClassical(ipl_stencil_list_standard,
                                                              mid_level, top_level, pre_assign=iadd)


    ipl_low_to_mid = InterpolationByStencilForLevelsClassical(ipl_stencil_list_standard,
                                                              low_level, mid_level, pre_assign=iadd)

    # initialize top level
    top_level.arr[:] = 105.0
    # top_level.arr[:] = 0.0
    top_level.res[:] = 0.0
    top_level.rhs[:] = 0.0

    top_level.pad()
    mid_level.arr[:] = 0.0
    mid_level.res[:] = 0.0
    mid_level.rhs[:] = 0.0

    mid_level.pad()
    low_level.arr[:] = 0.0
    low_level.res[:] = 0.0
    low_level.rhs[:] = 0.0
    low_level.pad()
    mg_problem.fill_rhs(top_level)

    # laplace_stencil.modify_rhs(top_level)
    print("** rhs of the top level ** \n", top_level.rhs)
    print("**TopLevel before at initial value", top_level.arr)
    # we smooth in order to have something to restrict
    top_jacobi_smoother.relax(n_jacobi_pre)
    # print("TopLevel after smoothing: \n", top_level.arr)
    print("**TopLevel after "+str(n_jacobi_pre)+" jacob iterations **\n", top_level.arr)
    # compute residuum
    top_level.compute_residual(laplace_stencil)
    print("** residuum on top level **\n", top_level.restrict_out)

    rst_top_to_mid.restrict()
    print("** restriction onto the mid_level **\n")
    print("rhs of mid_level : \n", mid_level.restrict_in)
    print("** MidLevel after "+str(n_jacobi_pre)+" jacobi iterations **")
    mid_jacobi_smoother.relax(n_jacobi_pre)
    print(mid_level.mid)
    # print("residual  before computation : \n", mid_level.res)
    mid_level.compute_residual(laplace_stencil)
    print("** residual on mid_level **\n", mid_level.res)
    print("** restriction onto the low_level **")
    rst_mid_to_low.restrict()
    print("rhs of low_level : \n", low_level.rhs)
    low_direct_smoother.relax()
    print("** LowLevel after direct solve **\n", low_level.mid)
    # low_jacobi_smoother.relax(n_jacobi)
    # print("** LowLevel after"+str(n_jacobi)+"jacobi iterations **\n", low_level.mid)

    # here we are deep down
    print("** coarse grid correct the mid_level with the result of the low_level **")
    ipl_low_to_mid.eval()
    print("mid_level after correction : \n", mid_level.mid)
    mid_jacobi_smoother.relax(n_jacobi_post)
    print("** mid_level after "+str(n_jacobi_post)+" Jacob iterations ** \n", mid_level.mid)
    print("** coarse grid correct the top_level with the result of the mid_level **")
    ipl_mid_to_top.eval()
    print("top_level after correction : \n", top_level.mid)
    top_jacobi_smoother.relax(n_jacobi_post)
    temp = np.copy(top_level.mid)
    print("** top_level after "+str(n_jacobi_post)+"Jacob iterations ** \n", top_level.mid)
    laplace_stencil.modify_rhs(top_level)
    top_direct_solver.relax()
    print("** rhs of the top level ** \n", top_level.rhs)
    print("** as a reference the direct solution of TopLevel **\n", top_level.mid)
    print("** and the difference **\n", temp-top_level.mid)
