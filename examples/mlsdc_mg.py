# coding=utf-8
"""
.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
.. moduleauthor:: Dieter Moser <d.moser@fz-juelich.de>
"""
import numpy as np
from collections import OrderedDict

from pypint.utilities.logging import LOG, print_logging_message_tree, VERBOSITY_LVL1, SEPARATOR_LVL1, SEPARATOR_LVL2

LOG.info("%sSetting Up Multigrid Space Solver" % VERBOSITY_LVL1)
from pypint.plugins.multigrid.stencil import Stencil
laplace_stencil = Stencil(np.array([1.0, -2.0, 1.0]), None, 2)
LOG.info("%s  Laplace Discretization Stencil: %s" % (VERBOSITY_LVL1, laplace_stencil.arr))

LOG.info(SEPARATOR_LVL2)

LOG.info("%sSetting Up 1D Heat Equation" % VERBOSITY_LVL1)

LOG.info("%s  Setting Up Geometry" % VERBOSITY_LVL1)
geo = np.asarray([[0, 1]])

LOG.info("%s  Setting Up Boundary Functions" % VERBOSITY_LVL1)
boundary_types = ['dirichlet'] * 2
bnd_left_fnc = lambda x: 0.0
bnd_right_fnc = lambda x: 1.0
bnd_functions = [[bnd_left_fnc, bnd_right_fnc]]

num_points_mg_levels = OrderedDict()
num_points_mg_levels['finest'] = 4
# num_points_mg_levels['mid'] = 5
# num_points_mg_levels['base'] = 2
print_logging_message_tree(OrderedDict({'Points on Space Grid': num_points_mg_levels}))

from examples.problems.heat_equation import HeatEquation
problem = HeatEquation(dim=(num_points_mg_levels['finest'], 1),
                       time_end=0.2,
                       thermal_diffusivity=0.5,
                       # initial_value=np.array([[0.0], [0.0], [1.0], [0.0], [0.0]]),
                       rhs_function_wrt_space=lambda dof, tensor: 0.0,
                       boundary_functions=bnd_functions,
                       boundaries=boundary_types,
                       geometry=geo)

print_logging_message_tree(OrderedDict({'Problem': problem.print_lines_for_log()}))

LOG.info(SEPARATOR_LVL2)
LOG.info("%sSetting Up Multigrid Levels" % VERBOSITY_LVL1)
from pypint.plugins.multigrid.level import MultigridLevel1D
borders = np.array([3, 3])

fine_mg_level = MultigridLevel1D(num_points_mg_levels['finest'], mg_problem=problem, max_borders=borders, role='FL')
problem._mg_level = fine_mg_level
problem._mg_stencil = Stencil(np.array([problem.thermal_diffusivity, -2.0 * problem.thermal_diffusivity, problem.thermal_diffusivity]) / fine_mg_level.h**2)
problem._mg_stencil.grid = fine_mg_level.mid.shape
# LOG.debug("Sparse matrix: %s -> %s" % (problem._mg_stencil.sp_matrix.shape, problem._mg_stencil.sp_matrix.todense()))
# mid_mg_level = MultigridLevel1D(num_points_mg_levels['mid'], mg_problem=problem, max_borders=borders, role='ML')
# base_mg_level = MultigridLevel1D(num_points_mg_levels['base'], mg_problem=problem, max_borders=borders, role='CL')

LOG.info("%s  Levels" % VERBOSITY_LVL1)
LOG.info("%s    Top Level" % VERBOSITY_LVL1)
LOG.info("%s      h: %s" % (VERBOSITY_LVL1, fine_mg_level.h))
# LOG.info("%s    Middle Level" % VERBOSITY_LVL1)
# LOG.info("%s      h: %s" % (VERBOSITY_LVL1, mid_mg_level.h))
# LOG.info("%s    Base Level" % VERBOSITY_LVL1)
# LOG.info("%s      h: %s" % (VERBOSITY_LVL1, base_mg_level.h))

# LOG.info(SEPARATOR_LVL2)
# LOG.info("%sSetting Up Multigrid Smoothers" % VERBOSITY_LVL1)
# from pypint.plugins.multigrid.multigrid_smoother import SplitSmoother, DirectSolverSmoother
# define the smoother from the split smoother class on each level,
# where the last level is solved directly
# omega = 1/np.sqrt(2)
# omega = 0.5
# l_plus = np.asarray([0, -2.0/omega, 0])
# l_minus = np.asarray([1.0, -2.0*(1.0 - 1.0/omega), 1.0])
# top_jacobi_smoother = SplitSmoother(l_plus / fine_mg_level.h**2,
#                                     l_minus / fine_mg_level.h**2,
#                                     fine_mg_level)
# mid_jacobi_smoother = SplitSmoother(l_plus / mid_mg_level.h**2,
#                                     l_minus / mid_mg_level.h**2,
#                                     mid_mg_level)
# low_jacobi_smoother = SplitSmoother(l_plus / base_mg_level.h**2,
#                                     l_minus / base_mg_level.h**2,
#                                     base_mg_level)
# low_direct_smoother = DirectSolverSmoother(laplace_stencil, base_mg_level)

LOG.info(SEPARATOR_LVL2)
LOG.info("%sSetting Up Multigrid Level Transitions" % VERBOSITY_LVL1)
# from operator import iadd
# from pypint.plugins.multigrid.restriction import RestrictionByStencilForLevelsClassical
# from pypint.plugins.multigrid.interpolation import InterpolationByStencilForLevelsClassical
# center = np.asarray([0])
n_jacobi_pre = 1
n_jacobi_post = 1
# we define the Restriction operator
# rst_stencil = Stencil(np.asarray([0.25, 0.5, 0.25]))
# rst_top_to_mid = RestrictionByStencilForLevelsClassical(rst_stencil, fine_mg_level, mid_mg_level)
# rst_mid_to_low = RestrictionByStencilForLevelsClassical(rst_stencil, mid_mg_level, base_mg_level)

# and the interpolation operator
# ipl_stencil_list_standard = [(Stencil(np.asarray([1]), center), (1,)),
#                              (Stencil(np.asarray([0.5, 0.5]), center), (0,))]
#
# ipl_mid_to_top = InterpolationByStencilForLevelsClassical(ipl_stencil_list_standard,
#                                                           mid_mg_level, fine_mg_level, pre_assign=iadd)
#
# ipl_low_to_mid = InterpolationByStencilForLevelsClassical(ipl_stencil_list_standard,
#                                                           base_mg_level, mid_mg_level, pre_assign=iadd)

LOG.info(SEPARATOR_LVL2)
LOG.info("%sSetting Initial Values for MG Levels" % VERBOSITY_LVL1)
# initialize top level
fine_mg_level.arr[:] = 0.0
# top_level.arr[:] = 0.0
fine_mg_level.res[:] = 0.0
fine_mg_level.rhs[:] = 0.0
fine_mg_level.pad()

# mid_mg_level.arr[:] = 0.0
# mid_mg_level.res[:] = 0.0
# mid_mg_level.rhs[:] = 0.0
# mid_mg_level.pad()

# base_mg_level.arr[:] = 0.0
# base_mg_level.res[:] = 0.0
# base_mg_level.rhs[:] = 0.0
# base_mg_level.pad()

problem.fill_rhs(fine_mg_level)

LOG.info(SEPARATOR_LVL2)
LOG.info("%sSetting Up MLSDC Solver" % VERBOSITY_LVL1)
from pypint.multi_level_providers.multi_time_level_provider import MultiTimeLevelProvider
from pypint.multi_level_providers.level_transition_providers.time_transition_provider import TimeTransitionProvider
from pypint.integrators.sdc_integrator import SdcIntegrator

base_mlsdc_level = SdcIntegrator()
base_mlsdc_level.init(num_nodes=5)

fine_mlsdc_level = SdcIntegrator()
fine_mlsdc_level.init(num_nodes=7)

transitioner = TimeTransitionProvider(fine_nodes=fine_mlsdc_level.nodes, coarse_nodes=base_mlsdc_level.nodes)

ml_provider = MultiTimeLevelProvider()
ml_provider.add_coarse_level(fine_mlsdc_level)
ml_provider.add_coarse_level(base_mlsdc_level)
ml_provider.add_level_transition(transitioner, 0, 1)

from pypint.communicators import ForwardSendingMessaging
comm = ForwardSendingMessaging()

from pypint.solvers.ml_sdc import MlSdc
mlsdc = MlSdc(communicator=comm)
comm.link_solvers(previous=comm, next=comm)
comm.write_buffer(tag=(ml_provider.num_levels - 1), value=problem.initial_value, time_point=problem.time_start)

mlsdc.init(problem=problem, ml_provider=ml_provider)

LOG.info(SEPARATOR_LVL1)
LOG.info("%sInitialize Direct Space Solvers for Time Levels" % VERBOSITY_LVL1)
for time_level in range(0, ml_provider.num_levels):
    _integrator = ml_provider.integrator(time_level)
    for time_node in range(0, _integrator.num_nodes - 1):
        problem.initialize_direct_space_solver(time_level,
                                               (_integrator.nodes[time_node + 1] - _integrator.nodes[time_node]),
                                               fine_mg_level)

LOG.info(SEPARATOR_LVL1)
LOG.info("%sLaunching MLSDC with MG" % VERBOSITY_LVL1)
from pypint.solvers.cores import SemiImplicitMlSdcCore, ExplicitMlSdcCore
mlsdc.run(SemiImplicitMlSdcCore, dt=0.2)

print("RHS Evaluations: %d" % problem.rhs_evaluations)
