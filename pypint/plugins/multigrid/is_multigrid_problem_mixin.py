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


class IsMultiGridProblemMixin(object):
    """Provides functionality of a problem to be have multigrid as its space solver

    Contains every aspect of the Problem that has to be solved, like the stencil from which on may derive :math:`A_h`
    for each level.
    """

    valid_boundary_conditions = ['periodic', 'dirichlet']

    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        stencil : :py:class:`.Stencil`
            multigrid stencil specifying the discretization of the problem
        function : :py:class:`callable`
            function returning the space-and-time-dependent values for the right hand side for the space solver
        stencil_center : :py:class:`numpy.ndarray` or :py:class:`int`
            *(optional)*
            defaults to the arithmetic center of given stencil
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

        assert_named_argument('stencil', kwargs, types=Stencil, descriptor="Stencil", checking_obj=self)
        assert_named_argument('function', kwargs, types=callable, descriptor="RHS for space solver", checking_obj=self)

        # the Space tensor which is actually used
        self._act_space_tensor = None
        self._act_grid_distances = None

        # the points actually used
        self._act_npoints = None

        self._function = kwargs['function']
        self._stencil = kwargs['stencil']
        self._dim = self.stencil.arr.ndim
        self._shape = self.stencil.arr.shape

        if kwargs.get('stencil_center') is None:
            self._stencil_center = self.mid_of_stencil(self.stencil)
        else:
            self._stencil_center = kwargs['stencil_center']

        # check if boundary conditions are specified
        if kwargs.get('boundaries') is None:
            self._boundaries = ['periodic'] * self.dim
        elif isinstance(kwargs['boundaries'], str) \
                and kwargs['boundaries'] in IsMultiGridProblemMixin.valid_boundary_conditions:
            self._boundaries = [kwargs['boundaries']] * self.dim
        elif isinstance(kwargs['boundaries'], list):
            check = 0
            for bc in kwargs['boundaries']:
                if bc in IsMultiGridProblemMixin.valid_boundary_conditions:
                    check += 1
            if check == self.dim * 2:
                self._boundaries = kwargs['boundaries']
            else:
                LOG.warning('Boundary specifications are not valid, will use periodic boundaries for each dimension.')
                self._boundaries = ['periodic'] * self.dim
        else:
            LOG.warning('Boundary specifications are not valid, will use periodic boundaries for each dimension')
            self._boundaries = ['periodic'] * self.dim

        # assign according to the boundary conditions the right functions
        if kwargs.get('boundary_functions') is None:
            self._boundary_functions = [None] * self.dim
        else:
            assert_is_instance(kwargs['boundary_functions'], list, descriptor="Boundary Functions", checking_obj=self)
            check = 0
            assert_condition(len(kwargs['boundary_functions']) == self.dim,
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
            self._geometry = np.asarray([[0, 1]] * self.dim)
        else:
            assert_is_instance(kwargs['geometry'], np.ndarray, descriptor="Geometry", checking_obj=self)
            assert_condition(len(kwargs["geometry"].shape) == 2, ValueError,
                             message="Numpy array has the wrong dimensions", checking_obj=self)
            assert_condition(kwargs['geometry'].shape[0] == self.dim and kwargs['geometry'].shape[1] == 2,
                             ValueError,
                             message="Numpy array has a wrong shape", checking_obj=self)
            self._geometry = kwargs['geometry']

        self._mg_system_matrices = {}

    @property
    def stencil(self):
        return self._stencil

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
    def act_grid_distances(self):
        """Getter for the current grid distances
        """
        return self._act_grid_distances

    @property
    def geometry(self):
        """Getter for the geometry
        """
        return self._geometry

    def mg_system_matrix(self, phi_of_time):
        _phi_of_time_shape = phi_of_time.shape
        if self._mg_system_matrices.get(_phi_of_time_shape) is None:
            LOG.debug("MG System Matrix not found for shape %s. Computing." % _phi_of_time_shape)
            self.stencil.to_sparse_matrix(_phi_of_time_shape, format='csr')
        return self._mg_system_matrices[_phi_of_time_shape]

    def mg_solve(self, next_x, method='direct', **kwargs):
        """Runs the multigrid solver

        Parameters
        ----------
        method : :py:class:`str`
            ``mg``
                for full multigrid cycles; additional keyword arguments passed to the multigrid solver can be given
            ``direct``
                for using the a predefined multigrid smoother as a direct solver via :py:class:`.DirectSolverSmoother`;
                defaults to ``direct``

        Raises
        ------
        ValueError
            if given ``method`` is not valid

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
            assert_named_argument('mg_level', kwargs, types=IMultigridLevel, descriptor="Multigrid Level",
                                  checking_obj=self)
            solver = DirectSolverSmoother(self.stencil, kwargs['mg_level'])
            return solver.relax()
        else:
            raise ValueError("Unknown method: '%s'" % method)

    def mg_time_evaluate(self, time, phi_of_time, partial=None, **kwargs):
        """Multigrid implementation for evaluating the right hand side with respect to the time solver
        """
        return self.mg_system_matrix(phi_of_time).dot(phi_of_time.flatten()).reshape(phi_of_time.shape)

    def mid_of_stencil(self, stencil):
        return np.floor(np.asarray(stencil.arr.shape) * 0.5)

    def construct_space_tensor(self, number_of_points_list, set_act=True):
        """Constructs the Spacetensor which is important for the evaluation in the case of Dirichlet boundary conditions

        Parameters
        ----------
        number_of_points_list : :py:class:`int` or :py:class:`numpy.ndarray`
            Number of points which will be distributed equiv-spaced on the grid
        """
        if isinstance(number_of_points_list, (int, float, complex)):
            npoints = int(number_of_points_list)
            LOG.debug("Your number %s was modified to %s" % (number_of_points_list, npoints))
            assert_condition(npoints > max(self._shape), ValueError,
                             message="Not enough points for the stencil", checking_obj=self)
            npoints = np.asarray([npoints] * self.dim)
        elif isinstance(number_of_points_list, np.ndarray):
            assert_condition(len(number_of_points_list.shape) == 1 and number_of_points_list.size == self.dim,
                             ValueError, message="The number_of_points list is wrong", checking_obj=self)
            npoints = np.floor(number_of_points_list)
        else:
            raise ValueError("Wrong number of points list")

        # first we assign the memory using numpy
        # spt(npoints,dim)
        self._act_npoints = npoints
        lspc = []
        for i in range(self.dim):
            lspc.append(np.linspace(self._geometry[i, 0], self._geometry[i, 1], npoints[i]))
        if self.dim > 1:
            space_tensor = np.asarray(np.meshgrid(*lspc))
        else:
            space_tensor = np.linspace(self._geometry[0, 0], self._geometry[0, 1], npoints)
        if set_act:
            self._act_space_tensor = space_tensor
            self._act_grid_distances = []
            zero_point = tuple([0] * self.dim)
            for i in range(self.dim):
                diff_point = tuple([0] * i + [1] + [0] * (self.dim - i - 1))
                self._act_grid_distances.append(space_tensor[diff_point] - space_tensor[zero_point])
            self._act_grid_distances = np.asarray(self._act_grid_distances)

        return space_tensor

    def fill_rhs(self, level):
        """Fills the rhs of an level
        """
        if level.space_tensor is None:
            level.space_tensor = self.construct_space_tensor(list(level.mid.shape))
        level.rhs[:] = self._function(level.mid, level.space_tensor)

    def eval_f(self, u=None, function=None, space_tensor=None):
        """Evaluates the right hand side with the actual space tensor, and the current :math:`u`.
        """
        assert_condition(self._act_space_tensor is not None, ValueError,
                         message="A current space tensor is needed", checking_obj=self)

        if function is None:
            if u is None:
                return self._function(self._act_space_tensor)
            else:
                assert_is_instance(u, np.ndarray, "u is not an numpy array")
                assert_condition(u.shape == self._act_space_tensor[1].shape,
                                 "u has the wrong shape", self)
                return self._function(u, self._act_space_tensor)
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
    """Checking whether

    """
    assert_is_instance(problem, IProblem,
                       message="It needs to be a problem to be a Multigrid problem.",
                       checking_obj=checking_obj)
    return isinstance(problem, IsMultiGridProblemMixin)


__all__ = ['problem_is_multigrid_problem', 'MultiGridProblem']
