# coding=utf-8
import numpy as np
import sys
print(sys.path)
from pypint.plugins.multigrid.multigrid_problem import MultiGridProblem
from pypint.plugins.multigrid.multigrid_level_provider import MultiGridLevelProvider
from pypint.plugins.multigrid.multigrid_solution import MultiGridSolution
from pypint.plugins.multigrid.level import MultigridLevel1D
from pypint.plugins.multigrid.multigrid_smoother import SplitSmoother, DirectSolverSmoother, WeightedJacobiSmoother
from pypint.utilities import assert_is_callable, assert_is_instance, assert_condition
from pypint.plugins.multigrid.stencil import Stencil
from pypint.plugins.multigrid.interpolation import InterpolationByStencilListIn1D, InterpolationByStencilForLevels, InterpolationByStencilForLevelsClassical
from pypint.plugins.multigrid.restriction import RestrictionStencilPure, RestrictionByStencilForLevels, RestrictionByStencilForLevelsClassical
from operator import iadd,add

def print_all(level):
    print("*** LevelPrint *** : 0x%x" % id(level))
    print("\tArr \n", level.arr)
    print("\tMid \n", level.mid)
    print("\tRhs \n", level.rhs)
    print("\tRes \n", level.res)


if __name__ == '__main__':
    laplace_array = np.asarray([1.0, -2.0, 1.0])
    laplace_stencil = Stencil(np.asarray([1, -2, 1]), None, 2)

    # preparing MG_Problem

    geo = np.asarray([[0, 1]])
    boundary_type = ["dirichlet"]*2
    left_f = lambda x: 1.0
    right_f = lambda x: 1.0
    boundary_functions = [[left_f, right_f]]

    def stupid_f(*args, **kwargs):
        return 0.0

    mg_problem = MultiGridProblem(laplace_stencil,
                                  stupid_f,
                                  boundary_functions=boundary_functions,
                                  boundaries=boundary_type, geometry=geo)
    borders = np.asarray([2, 2])

    top_level = MultigridLevel1D(11, mg_problem=mg_problem,
                                 max_borders=borders, role="FL")

    mid_level = MultigridLevel1D(5, mg_problem=mg_problem,
                                 max_borders=borders, role="ML")

    low_level = MultigridLevel1D(2, mg_problem=mg_problem,
                                 max_borders=borders, role="CL")

    omega = 1.0
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


    top_stencil = Stencil(laplace_array / top_level.h**2)
    mid_stencil = Stencil(laplace_array / mid_level.h**2)
    low_stencil = Stencil(laplace_array / low_level.h**2)

    low_direct_smoother = DirectSolverSmoother(low_stencil, low_level)
    center = np.asarray([0])

    # restriction
    rst_stencil = Stencil(np.asarray([0.25, 0.5, 0.25]))
    rst_top_to_mid = RestrictionByStencilForLevelsClassical(top_level, mid_level, rst_stencil)
    rst_mid_to_low = RestrictionByStencilForLevelsClassical(mid_level, low_level, rst_stencil)

    # ipl
    ipl_stencil_list_standard = [(Stencil(np.asarray([1]), center), (1,)),
                                   (Stencil(np.asarray([0.5, 0.5]), center), (0,))]

    ipl_mid_to_top = InterpolationByStencilForLevelsClassical(mid_level, top_level, ipl_stencil_list_standard,
                                                              pre_assign=iadd)

    ipl_low_to_mid = InterpolationByStencilForLevelsClassical(low_level, mid_level, ipl_stencil_list_standard,
                                                              pre_assign=iadd)

    n_pre = 1
    n_post = 1
    print("\t \t --Nothing done--\n")
    print_all(top_level)

    top_level.mid[:] = 1
    top_level.pad()
    mg_problem.fill_rhs(top_level)
    top_stencil.modify_rhs(top_level)
    print("\t \t --Initialisation of top level --\n")
    print_all(top_level)

    top_jacobi_smoother.relax(n_pre)

    print("\t\t --After %d Jacobi iterations --" % n_pre)
    print_all(top_level)

    top_level.compute_residual(top_stencil)

    print("\t\t --After the computation of the residual--")
    print_all(top_level)

    rst_top_to_mid.restrict()

    print("\t\t --After restriction to mid level")
    print_all(mid_level)

    mid_jacobi_smoother.relax(n_pre)

    print("\t\t --After %d Jacobi iterations --" % n_pre)
    print_all(mid_level)

    mid_level.compute_residual(mid_stencil)

    print("\t\t --After the computation of the residual--")
    print_all(mid_level)

    rst_mid_to_low.restrict()
    print("\t\t --After restriction to low level")
    print_all(low_level)

    low_direct_smoother.relax()

    print("\t\t --After direct solve --")
    print_all(low_level)

    ipl_low_to_mid.eval()

    print("\t\t --After interpolation to mid level")
    print_all(mid_level)

    mid_jacobi_smoother.relax(n_post)

    print("\t\t --After %d Jacobi iterations --" % n_post)
    print_all(mid_level)

    ipl_mid_to_top.eval()

    print("\t\t --After interpolation to top level")
    print_all(top_level)

    mid_jacobi_smoother.relax(n_post)

    print("\t\t --After %d Jacobi iterations --" % n_post)
    print_all(top_level)

    # real solution
    sol_level = MultigridLevel1D(11, mg_problem=mg_problem,
                                 max_borders=borders, role="FL")
    sol_level.mid[:] = 0.
    sol_level.pad()
    mg_problem.fill_rhs(sol_level)
    top_stencil.modify_rhs(sol_level)

    sol_direct_smoother = DirectSolverSmoother(top_stencil, sol_level)
    print("\t\t --real solution--")
    sol_direct_smoother.relax()
    print_all(sol_level)
