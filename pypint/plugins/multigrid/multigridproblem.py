    # coding=utf-8
"""
MultigridProblem
"""

import numpy as np
import scipy.signal as sig
from pypint.utilities import assert_is_callable, assert_is_instance, assert_condition
from pypint.plugins.multigrid.stencil import Stencil


def stupid_func(x, *args, **kwargs):
    """Duh I am stupid"""
    return 1.0


class MultiGridProblem(object):
    """
    Summary
    _______
    Contains every aspect of the Problem that has to be solved,
    like the stencil from which on may derive A_h for each level. A_h=b
    """
    def __init__(self, stencil, function, stencil_center=None, **kwargs):
        # the Space tensor which is actually used
        self._act_space_tensor = None
        self._act_grid_distances = None
        # the points actually used
        self._act_npoints = None
        self.valid_boundary_conditions = ['periodic', 'dirichlet']
        assert_is_instance(stencil, Stencil,
                           "The stencil has to be a Stencil", self)

        assert_is_callable(function,
                           "The object is not callable", self)
        self._function = function
        self._stencil = stencil
        self._numeric_type = np.float
        self._dimension = stencil.arr.ndim
        self._shape = stencil.arr.shape

        if stencil_center is None:
            self._stencil_center = self.mid_of_stencil(stencil)
        else:
            self._stencil_center = stencil_center
        # check if boundary conditions are specified
        if kwargs.get('boundaries') is None:
            self._boundaries = ["periodic"]*self._dimension
        elif isinstance(kwargs["boundaries"], str) and\
                        kwargs["boundaries"] in self.valid_boundary_conditions:
            self._boundaries = [kwargs['boundaries']]*self._dimension
        elif isinstance(kwargs["boundaries"], list):
            check = 0
            for bc in kwargs["boundaries"]:
                if bc in self.valid_boundary_conditions:
                    check += 1
            if check == self._dimension*2:
                self._boundaries = kwargs["boundaries"]
            else:
                print('Boundary specifications are not valid,',
                      'will use periodic boundaries for each dimension')
                self._boundaries = ["periodic"] * self._dimension
        else:
            print('Boundary specifications are not valid,',
                  'will use periodic boundaries for each dimension')
            self._boundaries = ["periodic"] * self._dimension
        # assign according to the boundary conditions the right functions
        if kwargs.get("boundary_functions") is None:
            self._boundary_functions = [None]*self._dimension
        elif isinstance(kwargs["boundary_functions"], list):
            check = 0
            assert_condition(len(kwargs["boundary_functions"]) is self._dimension,
                             "Not enough function tupel", self)
            for ftpls in kwargs["boundary_functions"]:
                if ftpls is "dirichlet":
                    assert_condition(isinstance(ftpls, list),
                                     "Dirichlet function list not available",
                                     self)
                    assert_condition(len(ftpls) is 2,
                                     "Wrong number of functions", self)
                    assert_is_callable(ftpls[0], "Not a function", self)
                    assert_is_callable(ftpls[1], "Not a function", self)
                check += 1
            self._boundary_functions = kwargs["boundary_functions"]
        else:
            assert_condition(True, "This shouldn't happen", self)
        # construct or save the geometry
        if kwargs.get("geometry") is None:
            self._geometry = np.asarray([[0, 1]]*self._dimension)
        elif isinstance(kwargs["geometry"], np.ndarray):
            assert_condition(len(kwargs["geometry"].shape) == 2,
                             "Numpy array has the wrong dimensions", self)
            assert_condition(kwargs["geometry"].shape[0] == self._dimension and
                             kwargs["geometry"].shape[1] == 2,
                             "Numpy array has a wrong shape", self)
            self._geometry = kwargs["geometry"]
        else:
            raise ValueError("Geometry is not a numpy array")

    @property
    def dimension(self):
        """
        Summary
        -------
        Getter for the dimension
        """
        return self._dimension

    @property
    def boundaries(self):
        """
        Summary
        -------
        Getter for the boundarietypes
        """
        return self._boundaries

    @property
    def boundary_functions(self):
        """
        Summary
        -------
        Getter for the boundary functions
        """
        return self._boundary_functions

    @property
    def act_grid_distances(self):
        """
        Summary
        -------
        Getter for the current grid distances
        """
        return self._act_grid_distances

    @property
    def geometry(self):
        """
        Summary
        -------
        Getter for the geometry
        """
        return self._geometry

    def mid_of_stencil(self, stencil):
        return np.floor(np.asarray(stencil.arr.shape)*0.5)

    def construct_space_tensor(self, number_of_points_list, set_act = True):
        """
        Summary
        -------
        Constructs the Spacetensor which is important for the evaluation
        in the case of Dirichlet boundary conditions.
        Parameters
        ----------
        number_of_points_list : integer or numpy.ndarray
            Number of points which will be distributed equiv-spaced on the grid
        """

        if isinstance(number_of_points_list, (int, float, complex)):
            npoints = int(number_of_points_list)
            print("Your number " + str(number_of_points_list) +
                  " was modified to "+ str(npoints))
            assert_condition(npoints > max(self._shape),
                             "Not enough points for the stencil", self)
            npoints = np.asarray([npoints] * self._dimension)
        elif isinstance(number_of_points_list, np.ndarray):
            assert_condition(len(number_of_points_list.shape) == 1 and
                             number_of_points_list.size == self._dimension,
                             "The number_of_points list is wrong", self)
            npoints = np.floor(number_of_points_list)
        else:
            raise ValueError("Wrong number of points list")

        # first we assign the memory using numpy
        # spt(npoints,dim)
        self._act_npoints = npoints
        lspc = []
        for i in range(self._dimension):
            lspc.append(np.linspace(self._geometry[i, 0], self._geometry[i, 1],
                                    npoints[i]))
        if self._dimension > 1:
            space_tensor = np.asarray(np.meshgrid(*lspc))
        else:
            space_tensor = np.linspace(self._geometry[0, 0],
                                       self._geometry[0, 1],
                                       npoints)
        if set_act:
            self._act_space_tensor = space_tensor
            self._act_grid_distances = []
            zero_point = tuple([0]*self._dimension)
            for i in range(self._dimension):
                diff_point = tuple([0]*i+[1]+[0]*(self._dimension - i - 1))
                self._act_grid_distances.append(- space_tensor[zero_point]
                                                + space_tensor[diff_point])
            self._act_grid_distances = np.asarray(self._act_grid_distances)
        return space_tensor

    def fill_rhs(self, level):
        """
        Fills the rhs of an level
        """
        if level.space_tensor is None:
            level.space_tensor = self.construct_space_tensor(list(level.mid.shape))
        level.rhs[:] = self._function(level.mid, level.space_tensor)

    def eval_f(self, u=None, function=None, space_tensor = None):
        """
        Summary
        -------
        Evaluates the right hand side with the actual space tensor, and
        the current u.
        """
        assert_condition(self._act_space_tensor is not None,
                         "A current space tensor is needed", self)

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


    def pad_for_stencil(self, u, up=None, stencil=None):
        """
        Does the padding of u according to the boundaries and the stencil
        """
        assert_condition(self._act_space_tensor is not None,
                         "Not ready yet, current space tensor is needed use " +
                         "construct_space_tensor", self)
        if stencil is None:
            A = self._stencil
        else:
            assert_is_instance(stencil, np.ndarray)
            assert_condition(len(stencil.shape) == self._dimension,
                             "Not a valid stencil", self)
            A = stencil
        # vllt. würde es besser klappen, wenn die platz fuer eine padded
        # version schon vorliegen wuerde
        # the next step is to construct a padded version of the
        # "vector" u
        # the stencil indicates
        borders = [[0.0, 0.0]]*self._dimension
        for i in range(self._dimension):
            left = stencil.shape[i] - self._stencil_center[i] - 1
            right = stencil.shape[i] - left - 1
            borders[self._dimension - i - 1] = [left, right]
        # the following example shows how np.shape is organized
        #   >> d=np.arange(12).reshape(3,4)
        #   >> d.shape
        #     .. ( 3, 4)
        # this is inverse to the organisation of borders
        #  it holds  borders[n-i-1] = shape[i]
        # this borders may be used to produce a tensor in the right shape
        if up is None:
            u_pad = np.zeros(np.asarray(u.shape) + np.asarray(stencil.shape)-1)
        else:
            u_pad = up
        # now we embed u into u_pad
        # first we need to create the list of indices lists to
        # indicate where to embed u

        where_to_embed_u = []
        k = 0
        for i in borders:
            where_to_embed_u.append(np.arange(i[0], u_pad.shape(k)-i[1]))
            k += 1

        u_pad[tuple(where_to_embed_u)] = u

        # now we fill the border areas.
        # we will write simply for each  dimension another algorithm
        # a general algorithm for each direction would be great

        if self._dimension == 1:
            if self._boundaries[0] == 'periodic':
                #  left side
                u_pad[:borders[0][0]] = u[-borders[0][0]:]
                #  right side
                u_pad[-borders[0][1]:] = u[:borders[0][1]]
            else:
                fl = self._boundary_functions[0][0]
                fr = self._boundary_functions[0][1]
                # left from border
                l_f_b = np.linspace(-borders[0][0], -1, borders[0][0]) *\
                            self._act_grid_distances[0] +\
                            self._geometry[0][0]
                # right_from_border
                r_f_b = np.linspace(1, borders[0][1], borders[0][1]) *\
                            self._act_grid_distances[0] +\
                            self._geometry[0][1]
                #  left side
                u_pad[:borders[0][0]] = fl(l_f_b)
                #  right side
                u_pad[-borders[0][1]:] = fr(r_f_b)
        elif self._dimension == 2:
            # u_pad =
            # c1|   e1  | c2
            #---|-------|----
            # e4|   u   | e2
            #---|-------|----
            # c4|   e3  | c3
            #
            # each area has to be treated separately

            # first we take care of the certain cases

            if self._boundaries[0] == 'periodic':
                # this affects the areas e1 and e3
                # e1 first
                u_pad[:borders[1][0], borders[0][0]:-borders[0][1]] = \
                    u[-borders[1][0]:, :]
                # e3 next
                u_pad[-borders[1][1]:, borders[0][0]:-borders[0][1]] = \
                    u[:borders[1][1], :]
            else:
                # dirichlet
                # e1
                f_e1 = self._boundary_functions[0][0]

                e1_y = np.linspace(-borders[1][0], -1, borders[1][0]) *\
                            self._act_grid_distances[1] +\
                            self._geometry[1][0]
                e1_x = self._act_space_tensor[0][0]
                # e3_x = e1_x
                e_1 = np.meshgrid(e1_x, e1_y)
                u_pad[:borders[1][0], borders[0][0]:-borders[0][1]] = f_e1(e_1)
                # e3
                f_e3 = self._boundary_functions[0][1]
                e3_y = np.linspace(1, borders[1][1], borders[1][1]) *\
                            self._act_grid_distances[1] +\
                            self._geometry[1][1]
                e_3 = np.meshgrid(e1_x, e3_y)

                u_pad[-borders[1][1]:, borders[0][0]:-borders[0][1]] = \
                    f_e3(e_3)

            if self._boundaries[1] == 'periodic':
                # e4
                u_pad[borders[1][0]:-borders[1][1], :borders[0][0]] = \
                    u[:, -borders[0][0]:]
                # e2
                u_pad[borders[1][0]:-borders[1][1], -borders[0][1]:] = \
                    u[:, :borders[0][1]]
            else:
                # dirichlet
                # e4
                f_e4 = self._boundary_functions[1][0]

                e4_x = np.linspace(-borders[0][0], -1, borders[0][0]) *\
                            self._act_grid_distances[0] +\
                            self._geometry[0][0]

                e4_y = self._act_space_tensor[1][:, 0]
                # e2_y = e4_y
                e_4 = np.meshgrid(e4_x, e4_y)
                u_pad[borders[1][0]:-borders[1][1], :borders[0][0]] = f_e4(e_4)
                # e2
                f_e2 = self._boundary_functions[1][1]

                e2_x = np.linspace(1, borders[0][1], borders[0][1]) *\
                            self._act_grid_distances[0] +\
                            self._geometry[0][1]
                e_2 = np.meshgrid(e2_x, e4_y)
                u_pad[borders[1][0]:-borders[1][0], -borders[0][1]:] = \
                    f_e2(e_2)

            # next step is to treat the corners c_1 to c_2
            # therefore we have to work through the 4 cases
            # both periodic, both dirichlet, and the two mixed case

            if self._boundaries[0] == 'periodic' and self._boundaries[1] == 'periodic':
                # c1
                u_pad[:borders[1][0], :borders[0][0]] = \
                    u[-borders[1][0]:, -borders[0][0]:]
                # c2
                u_pad[:borders[1][0], -borders[0][1]:] = \
                    u[-borders[1][0]:, :borders[0][1]]
                #c3
                u_pad[-borders[1][1]:, -borders[0][1]:] = \
                    u[:borders[1][1], :borders[0][1]]
                #c4
                u_pad[-borders[1][1]:, :borders[0][0]] = \
                    u[:borders[1][1], -borders[0][0]:]
            elif self._boundaries[0] == 'dirichlet' and self._boundaries[1] == 'dirichlet':
                # c1
                c1 = np.meshgrid(e4_x, e1_y)
                u_pad[:borders[1][0], :borders[0][0]] = \
                    0.5 * (f_e1(c1) + f_e4(c1))
                # c2
                c2 = np.meshgrid(e2_x, e1_y)
                u_pad[:borders[1][0], -borders[0][1]:] = \
                    0.5 * (f_e1(c2) + f_e2(c2))
                #c3
                c3 = np.meshgrid(e2_x, e3_y)
                u_pad[-borders[1][1]:, -borders[0][1]:] = \
                    0.5 * (f_e2(c3) + f_e3(c3))
                #c4
                c4 = np.meshgrid(e4_x, e3_y)
                u_pad[-borders[1][1]:, :borders[0][0]] = \
                    0.5 * (f_e3(c4) + f_e4(c4))
            elif self._boundaries[0] == 'dirichlet' and self._boundaries[1] == 'periodic':
                # c1
                u_pad[:borders[1][0], :borders[0][0]] = \
                    u_pad[-borders[1][1]-borders[1][0]:-borders[1][1], :borders[0][0]]
                # c2
                u_pad[:borders[1][0], -borders[0][1]:] = \
                    u_pad[-borders[1][1]-borders[1][0]:-borders[1][1], -borders[0][1]:]
                #c3
                u_pad[-borders[1][1]:, -borders[0][1]:] = \
                    u_pad[borders[1][0]:borders[1][0]+borders[1][1], -borders[0][0]:]
                #c4
                u_pad[-borders[1][1]:, :borders[0][0]] = \
                    u_pad[borders[1][0]:borders[1][0]+borders[1][1], :borders[0][0]]
            elif self._boundaries[0] == 'periodic' and self._boundaries[1] == 'dirichlet':
                # c1
                u_pad[:borders[1][0], :borders[0][0]] = \
                    u_pad[:borders[1][0], -borders[0][0]-borders[0][1]:-borders[0][1]]
                # c2
                u_pad[:borders[1][0], -borders[0][1]:] = \
                    u_pad[:borders[1][0], borders[1][0]:borders[0][0]+borders[0][1]]
                #c3
                u_pad[-borders[1][1]:, -borders[0][1]:] = \
                    u_pad[-borders[1][1]:, -borders[0][0]-borders[0][1]:-borders[0][1]]
                #c4
                u_pad[-borders[1][1]:, :borders[0][0]] = \
                    u_pad[-borders[1][1]:, borders[1][0]:borders[0][0]+borders[0][1]]
        elif self._dimension > 2:
            print("Wer will den sowas . . .")
        # now we have a padded version with ghost cell in each direction

        return u_pad

        # and now we may use for example the convolve operator
        #    return sig.convolve(u_pad, A, 'valid')

