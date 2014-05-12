# coding=utf-8
"""IsMultigridProblemMixin

.. moduleauthor:: Dieter Moser <d.moser@fz-juelich.de>
.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
import numpy as np

from pypint.problems.i_problem import IProblem
from pypint.plugins.multigrid.stencil import Stencil
from pypint.plugins.multigrid.i_multigrid_level import IMultigridLevel
from pypint.plugins.multigrid.multigrid_smoother import DirectSolverSmoother
from pypint.utilities import assert_is_callable, assert_is_instance, assert_condition, assert_named_argument
from pypint.utilities.logging import LOG


class MultigridProblemMixin(object):
    """Provides functionality of a problem to have multigrid as its space solver

    Contains every aspect of the Problem that has to be solved, like the stencil from which on may derive :math:`A_h`
    for each level.
    """

    valid_boundary_conditions = ['periodic', 'dirichlet']

    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        rhs_function_wrt_space : :py:class:`callable`
            function returning the space-dependent values for the right hand side as used by the space solver
        boundaries : :py:class:`None` or :py:class:`list` of :py:class:`str`
            *(optional)*
            defaults to ``periodic`` for each dimension
        boundary_functions : :py:class:`None` or :py:class:`list` of :py:class:`callable`
            *(optional)*
            functions defined on the boundaries of the geometry
        geometry : :py:class:`None` or :py:class:`numpy.ndarray`
            *(optional)*
            specifying the dimension and extend of the geometry
        """
        assert_is_instance(self, IProblem, message="This Mixin is only valid for IProblems.", checking_obj=self)

        assert_named_argument('rhs_function_wrt_space', kwargs, descriptor="RHS for space solver", checking_obj=self)
        assert_is_callable(kwargs['rhs_function_wrt_space'], descriptor="RHS for space solver", checking_obj=self)

        self._rhs_function_wrt_space = kwargs['rhs_function_wrt_space']

        # check if boundary conditions are specified
        if kwargs.get('boundaries') is None:
            self._boundaries = ['periodic'] * len(self.spacial_dim)
        elif isinstance(kwargs['boundaries'], str) \
                and kwargs['boundaries'] in MultigridProblemMixin.valid_boundary_conditions:
            self._boundaries = [kwargs['boundaries']] * len(self.spacial_dim)
        elif isinstance(kwargs['boundaries'], list):
            check = 0
            for bc in kwargs['boundaries']:
                if bc in MultigridProblemMixin.valid_boundary_conditions:
                    check += 1
            if check == len(self.spacial_dim) * 2:
                self._boundaries = kwargs['boundaries']
            else:
                LOG.warning('Boundary specifications are not valid, will use periodic boundaries for each dimension.')
                self._boundaries = ['periodic'] * len(self.spacial_dim)
        else:
            LOG.warning('Boundary specifications are not valid, will use periodic boundaries for each dimension')
            self._boundaries = ['periodic'] * len(self.spacial_dim)

        # assign according to the boundary conditions the right functions
        if kwargs.get('boundary_functions') is None:
            self._boundary_functions = [None] * len(self.spacial_dim)
        else:
            assert_is_instance(kwargs['boundary_functions'], list, descriptor="Boundary Functions", checking_obj=self)
            check = 0
            assert_condition(len(kwargs['boundary_functions']) == len(self.spacial_dim),
                             ValueError, message="Not enough boundary functions given.", checking_obj=self)

            for ftpls in kwargs['boundary_functions']:
                if ftpls is 'dirichlet':
                    assert_is_instance(ftpls, list, message="Dirichlet function list not available", checking_obj=self)
                    assert_condition(len(ftpls) == 2,
                                     ValueError, message="Wrong number of functions", checking_obj=self)
                    assert_is_callable(ftpls[0], "Not a function", self)
                    assert_is_callable(ftpls[1], "Not a function", self)
                check += 1
            self._boundary_functions = kwargs['boundary_functions']

        # construct or save the geometry
        if kwargs.get('geometry') is None:
            self._geometry = np.asarray([[0, 1]] * len(self.spacial_dim))
        else:
            assert_is_instance(kwargs['geometry'], np.ndarray, descriptor="Geometry", checking_obj=self)
            assert_condition(len(kwargs["geometry"].shape) == 2, ValueError,
                             message="Numpy array has the wrong dimensions", checking_obj=self)
            assert_condition(kwargs['geometry'].shape[0] == len(self.spacial_dim) and kwargs['geometry'].shape[1] == 2,
                             ValueError,
                             message="Numpy array has a wrong shape", checking_obj=self)
            self._geometry = kwargs['geometry']

        self._rhs_space_operators = {}

        # the Space tensor which is actually used
        self._act_space_tensor = None
        self._act_grid_distances = None
        # the points actually used
        self._act_npoints = None

    def evaluate_wrt_space(self, **kwargs):
        """
        Parameters
        ----------
        values : :py:class:`numpy.ndarray`
        """
        assert_named_argument('values', kwargs, types=np.ndarray, descriptor="Values", checking_obj=self)
        return self.get_rhs_space_operators('default')\
                    .dot(kwargs['values'].flatten())\
                    .reshape(kwargs['values'].shape)

    @property
    def rhs_function_wrt_space(self):
        return self._rhs_function_wrt_space

    @rhs_function_wrt_space.setter
    def rhs_function_wrt_space(self, function):
        assert_is_callable(function, descriptor="Function of RHS w.r.t. Space", checking_obj=self)
        self._rhs_function_wrt_space = function

    @property
    def boundaries(self):
        """Getter for the boundarietypes
        """
        return self._boundaries

    @property
    def boundary_functions(self):
        """Getter for the boundary functions
        """
        return self._boundary_functions

    @property
    def geometry(self):
        """Getter for the geometry
        """
        return self._geometry

    def get_rhs_space_operators(self, delta_time):
        if self._rhs_space_operators.get(delta_time) is None:
            raise RuntimeError("MG System Matrix not found for delta time %s." % delta_time)
        return self._rhs_space_operators[delta_time]

    def set_rhs_space_operator(self, delta_time, operator='default'):
        self._rhs_space_operators[delta_time] = operator

    def mg_solve(self, next_x, method='direct', **kwargs):
        """Runs the multigrid solver

        This is where all the magic happens on each call of the space solver from the iterative time solver, i.e. on
        every iteration for each time-level in each sweep on each step.

        Parameters
        ----------
        method : :py:class:`str`
            defaults to ``direct``

            ``mg``
                for full multigrid cycles; additional keyword arguments passed to the multigrid solver can be given;
                additional arguments required:

                    ``mg_level``

            ``direct``
                for using the a predefined multigrid smoother as a direct solver via :py:class:`.DirectSolverSmoother`;
                additional arguments required:

                    ``mg_level``

                    ``stencil``

        Raises
        ------
        ValueError
            if given ``method`` is not one of ``mg`` or ``direct``

        Returns
        -------
        solution
        """
        if method == 'mg':
            assert_named_argument('mg_level', kwargs, types=IMultigridLevel, descriptor="Multigrid Level",
                                  checking_obj=self)
            LOG.debug("Using Multigrid as implicit space solver.")
            raise NotImplementedError("Full multigrid solver not yet plugged.")
        elif method == 'direct':
            if kwargs.get('solver') is None:
                assert_named_argument('mg_level', kwargs, types=IMultigridLevel, descriptor="Multigrid Level",
                                      checking_obj=self)
                assert_named_argument('stencil', kwargs, types=Stencil, descriptor="MG Stencil", checking_obj=self)
                solver_function = DirectSolverSmoother(kwargs['stencil'], kwargs['mg_level']).relax
            else:
                solver_function = kwargs['solver']
            LOG.debug("next_x.shape: %s" % (next_x.shape))
            return solver_function(next_x)
        else:
            raise ValueError("Unknown method: '%s'" % method)

    def construct_space_tensor(self, number_of_points_list, stencil=None):
        """Constructs the Spacetensor which is important for the evaluation in the case of Dirichlet boundary conditions

        Parameters
        ----------
        number_of_points_list : :py:class:`int` or :py:class:`numpy.ndarray`
            Number of points which will be distributed equiv-spaced on the grid
        """
        if isinstance(number_of_points_list, (int, float, complex)):
            assert_is_instance(stencil, Stencil, descriptor="Stencil", checking_obj=self)
            npoints = int(number_of_points_list)
            LOG.debug("Your number %s was modified to %s" % (number_of_points_list, npoints))
            assert_condition(npoints > max(stencil.arr.shape), ValueError,
                             message="Not enough points for the stencil", checking_obj=self)
            npoints = np.asarray([npoints] * len(self.spacial_dim))
        elif isinstance(number_of_points_list, np.ndarray):
            assert_condition(len(number_of_points_list.shape) == 1 and number_of_points_list.size == len(self.spacial_dim),
                             ValueError, message="The number_of_points list is wrong", checking_obj=self)
            npoints = np.floor(number_of_points_list)
        else:
            raise ValueError("Wrong number of points list")

        # first we assign the memory using numpy
        # spt(npoints,dim)
        self._act_npoints = npoints
        lspc = []
        for i in range(len(self.spacial_dim)):
            lspc.append(np.linspace(self._geometry[i, 0], self._geometry[i, 1], npoints[i]))
        if len(self.spacial_dim) > 1:
            space_tensor = np.asarray(np.meshgrid(*lspc))
        else:
            space_tensor = np.linspace(self._geometry[0, 0], self._geometry[0, 1], npoints)

        return space_tensor

    def fill_rhs(self, level):
        """Fills the rhs of an level
        """
        if level.space_tensor is None:
            level.space_tensor = self.construct_space_tensor(list(level.mid.shape))
        level.rhs[:] = self.rhs_function_wrt_space(level.mid, level.space_tensor)

    def eval_f(self, u=None, function=None, space_tensor=None):
        """Evaluates the right hand side with the actual space tensor, and the current :math:`u`.
        """
        assert_condition(self._act_space_tensor is not None, ValueError,
                         message="A current space tensor is needed", checking_obj=self)

        if function is None:
            if u is None:
                return self.rhs_function_wrt_space(self._act_space_tensor)
            else:
                assert_is_instance(u, np.ndarray, "u is not an numpy array")
                assert_condition(u.shape == self._act_space_tensor[1].shape,
                                 "u has the wrong shape", self)
                return self.rhs_function_wrt_space(u, self._act_space_tensor)
        else:
            if u is None:
                assert_is_callable(function, "Function is not callable", self)
                return function(self._act_space_tensor)
            else:
                assert_is_instance(u, np.ndarray, "u is not an numpy array")
                assert_condition(u.shape == self._act_space_tensor[1].shape,
                                 "u has the wrong shape", self)
                return function(u, self._act_space_tensor)


def problem_is_multigrid_problem(problem, checking_obj=None):
    assert_is_instance(problem, IProblem,
                       message="It needs to be a problem to be a Multigrid problem.",
                       checking_obj=checking_obj)
    return isinstance(problem, MultigridProblemMixin)


__all__ = ['problem_is_multigrid_problem', 'MultigridProblemMixin']
