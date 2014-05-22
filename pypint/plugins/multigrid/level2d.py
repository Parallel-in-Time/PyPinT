# coding=utf-8

import numpy as np
import scipy.signal as sig
from pypint.utilities import assert_is_callable, assert_is_instance, \
                                     assert_condition
# from pypint.plugins.multigrid.multigrid_problem import MultigridProblem
# from pypint.plugins.multigrid.stencil import Stencil
from pypint.plugins.multigrid.i_multigrid_level import IMultigridLevel


class MultigridLevel2D(IMultigridLevel):
    """
    Summary
    -------
    Simple usage of an numpy array, which allows
    the trouble-free use of a padded numpy array.
    Every calculation is applied to the non-padded version.
    The boundaries are used, whenever a convolution is applied.

    One aspect of this class is the port concept for interpolation and restriction
            _________________
            |               |
    FL :    | Level         |
            |               |
            -----------------
            ipl_in      rst_out

            ipl_out    rst_in
            _________________
            |               |
    ML :    | Level         |
            |               |
            -----------------
            ipl_in      rst_out


            ipl_out    rst_in
            _________________
            |               |
    CL :    | Level         |
            |               |
            -----------------



    Examples
    --------

    """
    def __init__(self, shape, mg_problem=None, max_borders=np.ones((2, 2)), dtype=float, role="ML"):
        """
        Summary
        -------
        takes the physical MultiGridProblem and initialises with help of
        max_borders, and n_points an appropriate array
        """
        # the level should know its geometrical information, because it differs from level to level
        self.space_tensor = None
        # assertion block for obvious bugs
        assert_is_instance(shape, tuple, "shape has to be a tuple")
        assert_condition(len(shape) == 2, ValueError, "shape has to be of length 2")
        assert_condition(len(mg_problem.spacial_dim) == 2, ValueError, "mg_problem has the wrong dimension")
        assert_is_instance(max_borders, np.ndarray, "max borders has to be a numpy array")
        assert_condition(max_borders.shape == (2, 2), ValueError, "max borders has the wrong shape")

        forward_shape = (shape[0] + max_borders[0][1]+max_borders[0][0],
                         shape[1] + max_borders[1][1]+max_borders[1][0])
        self.arr = np.zeros(forward_shape, dtype=dtype)
        self.borders = max_borders
        self.dim = 2
        self._mg_problem = mg_problem



        #define the right slices
        self.sl_mid_x = slice(self.borders[0][0], -self.borders[0][1])
        self.sl_mid_y = slice(self.borders[1][0], -self.borders[1][1])
        self.sl_front_x = slice(None, self.borders[0][0])
        self.sl_end_x = slice(-self.borders[0][1], None)
        self.sl_front_y = slice(None, self.borders[1][0])
        self.sl_end_y = slice(-self.borders[1][1], None)

        #define the parts
        self.mid = self.arr.__array__()[self.sl_mid_y, self.sl_mid_x]
        self.north = self.arr.__array__()[self.sl_front_y, self.sl_mid_x]
        self.south = self.arr.__array__()[self.sl_end_y, self.sl_mid_x]
        self.west = self.arr.__array__()[self.sl_mid_y, self.sl_front_x]
        self.east = self.arr.__array__()[self.sl_mid_y, self.sl_end_x]
        # north_east
        self.ne = self.arr.__array__()[self.sl_front_y, self.sl_end_x]
        # north_west
        self.nw = self.arr.__array__()[self.sl_front_y, self.sl_front_x]
        # south_east
        self.se = self.arr.__array__()[self.sl_end_y, self.sl_end_x]
        # south_west
        self.sw = self.arr.__array__()[self.sl_end_y, self.sl_front_x]

        # the first border points coincides with the geometrical border
        # that is why self.mid.size+1 is used instead of self.mid.size - 1

        self.h = np.asarray([0., 0.])
        self.h[0] = (self.mg_problem.geometry[0][1] - self.mg_problem.geometry[0][0]) / (self.mid.shape[0] + 1)

        self.h[1] = (self.mg_problem.geometry[1][1] - self.mg_problem.geometry[1][0]) / (self.mid.shape[1] + 1)
        # print("diff:", (self.mg_problem.geometry[0][1]
        #              - self.mg_problem.geometry[0][0]))
        # print("teiler:", (self.mid.shape[0] + 1))
        # define the tensors for each part
            #begin with the needed linspaces, to simplify the naming
            # l_spaces[axis][{0=front,1=mid,2=end}]
        self.l_spaces = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        for i in range(self.dim):
            # A-----B------------C----D

            # front : A -> B
            self.l_spaces[i][0] = np.arange(0, self.borders[i][0]) * \
                                    self.h[i] + self.mg_problem.geometry[i][0]
            # self.l_spaces[i][0] =
            # mid : B -> C
            self.l_spaces[i][1] = np.arange(1, self.mid.shape[i]+1) * \
                                    self.h[i] + self.mg_problem.geometry[i][0]
            # end : C -> D
            self.l_spaces[i][2] = np.linspace(0, self.borders[i][1], self.borders[i][1]) * \
                                    self.h[i] + self.mg_problem.geometry[i][1]
            # print("MultigridLevel2D linear spaces in direction "+str(i)+":")
            # print("front :\n", self.l_spaces[i][0])
            # print("mid :\n", self.l_spaces[i][1])
            # print("end :\n", self.l_spaces[i][2])
        # using this linear spaces we define space tensors for different parts
        self.mid_tensor = np.meshgrid(self.l_spaces[0][1], self.l_spaces[1][1])

        self.north_tensor = np.meshgrid(self.l_spaces[0][1], self.l_spaces[1][0])
        self.south_tensor = np.meshgrid(self.l_spaces[0][1], self.l_spaces[1][2])

        self.east_tensor = np.meshgrid(self.l_spaces[0][2], self.l_spaces[1][1])
        self.west_tensor = np.meshgrid(self.l_spaces[0][0], self.l_spaces[1][1])

        self.ne_tensor = np.meshgrid(self.l_spaces[0][2], self.l_spaces[1][0])
        self.nw_tensor = np.meshgrid(self.l_spaces[0][0], self.l_spaces[1][0])
        self.se_tensor = np.meshgrid(self.l_spaces[0][2], self.l_spaces[1][2])
        self.sw_tensor = np.meshgrid(self.l_spaces[0][0], self.l_spaces[1][2])

        # space for the rhs
        # self._rhs = np.copy(self.mid)
        self._rhs = np.zeros(self.mid.shape, dtype=dtype)
        lspc = []
        for i in range(self.dim):
            start = self._mg_problem.geometry[i][0] - self.h[i] * (max_borders[i][0] - 1)
            stop = self._mg_problem.geometry[i][1] + self.h[i] * (max_borders[i][1] - 1)
            lspc.append(np.linspace(start, stop, self.arr.shape[i]))
        self.space_tensor = np.asarray(np.meshgrid(*lspc))
        # set the interpolation and restriction ports according to the level which is used
        self.role = role
        # some place to store the residuum
        self.res = np.zeros(self.arr.shape)
        self.res_mid = self.res.__array__()[self.sl_mid_y, self.sl_mid_x]

        if role is "FL":
            # here we define the ports for the finest level
            self.interpolate_out = None
            self.interpolate_in = self.mid
            self.restrict_in = None
            self.restrict_out = self.res
            self.restriction_out_mid = self.res_mid
            # adjust boundary functions
            self.f_west = self._mg_problem.boundary_functions[0][0]
            self.f_east = self._mg_problem.boundary_functions[0][1]
            self.f_north = self._mg_problem.boundary_functions[1][0]
            self.f_south = self._mg_problem.boundary_functions[1][1]
        elif role is "ML":
            # here we define the ports for the mid level
            self.interpolate_out = self.arr
            self.interpolate_out_mid = self.mid
            self.interpolate_in = self.mid
            self.restrict_in = self.rhs
            self.restrict_out = self.res
            self.restriction_out_mid = self.res_mid

        elif role is "CL":
            # here we define the ports for the coarsest level
            self.interpolate_out = self.arr
            self.interpolate_out_mid = self.mid
            self.interpolate_in = None
            self.restrict_in = self.rhs
            self.restrict_out = None
        else:
            raise ValueError("MultiLevel has no role "+self.role)

        if role is not "FL":
            self.f_west = lambda x: 0.
            self.f_east = lambda x: 0.
            self.f_north = lambda x: 0.
            self.f_south = lambda x: 0.

        # in order to know if the rhs was modified
        self.modified_rhs = False
        self.mid_slice = (self.sl_mid_x, self.sl_mid_y)

    def adjust_references(self):
        #define the parts
        self.mid = self.arr.__array__()[self.sl_mid_y, self.sl_mid_x]
        self.north = self.arr.__array__()[self.sl_front_y, self.sl_mid_x]
        self.south = self.arr.__array__()[self.sl_end_y, self.sl_mid_x]
        self.west = self.arr.__array__()[self.sl_mid_y, self.sl_front_x]
        self.east = self.arr.__array__()[self.sl_mid_y, self.sl_end_x]
        # north_east
        self.ne = self.arr.__array__()[self.sl_front_y, self.sl_front_x]
        # north_west
        self.nw = self.arr.__array__()[self.sl_front_y, self.sl_end_x]
        # south_east
        self.se = self.arr.__array__()[self.sl_end_y, self.sl_end_x]
        # south_west
        self.sw = self.arr.__array__()[self.sl_end_y, self.sl_front_x]
        # and the residuum
        self.res_mid = self.res.__array__()[self.sl_mid_y, self.sl_mid_x]

    @property
    def mg_problem(self):
        """
        return MultiGridProblem
        """
        return self._mg_problem

    @property
    def rhs(self):
        return self._rhs

    @rhs.setter
    def rhs(self, value):
        self.modified_rhs = False
        self._rhs[:] = value

    def embed(self, ue):
        """
        Summary
        _______
        checks if u fits then embeds it

        Parameters
        ----------
        ue : ndarray
            numpy array to embed
        """
        assert_condition(ue.shape == self.mid.shape, ValueError,
                         "Array to embed has the wrong size")
        self.mid = ue

    def pad(self):
        """
        Summary
        -------
        Uses the information in Multigridproblems in order to
        pad the array.
        """

        # just dirichlet conditions

        if self.mg_problem.boundaries[0] is 'dirichlet' and self.mg_problem.boundaries[1] is 'dirichlet':

            self.north[:] = self.f_north(self.north_tensor)
            self.east[:] = self.f_east(self.east_tensor)
            self.south[:] = self.f_south(self.south_tensor)
            self.west[:] = self.f_west(self.west_tensor)
            self.ne[:] = self.f_north(self.ne_tensor) * 0.5 + self.f_east(self.ne_tensor) * 0.5
            self.nw[:] = self.f_north(self.nw_tensor) * 0.5 + self.f_west(self.nw_tensor) * 0.5
            self.se[:] = self.f_south(self.se_tensor) * 0.5 + self.f_east(self.se_tensor) * 0.5
            self.sw[:] = self.f_south(self.sw_tensor) * 0.5 + self.f_west(self.sw_tensor) * 0.5
        elif self.mg_problem.boundaries[0] is 'periodic' and self.mg_problem.boundaries[1] is 'periodic':
            self.east[:] = self.mid[:, self.sl_front_x]
            self.west[:] = self.mid[:, self.sl_end_x]
            self.north[:] = self.mid[self.sl_end_y, :]
            self.south[:] = self.mid[self.sl_front_y, :]
            self.ne[:] = self.mid[self.sl_end_y, self.sl_front_x]
            self.nw[:] = self.mid[self.sl_end_y, self.sl_end_x]
            self.se[:] = self.mid[self.sl_front_y, self.sl_front_x]
            self.sw[:] = self.mid[self.sl_front_y, self.sl_end_x]
        else:
            raise NotImplementedError("Bis jetzt sind nur Dirichlet Randbedingungen implementiert")


    def _evaluable_view(self, stencil, arr, offset=[0, 0]):
        """gives the right view of the array

        """
        if (stencil.b == self.borders).all():
            return self.arr
        else:
            slices = []
            for i in range(self.dim):
                slices.append(slice(self.borders[i][0] - stencil.b[i][0] + offset[0],
                                    -(self.borders[i][1] - stencil.b[i][1]) + offset[1]))

            return self.arr[tuple(slices)]


    def evaluable_view(self, stencil, offset=[0,0]):
        """gives the right view of the array

        """
        return self._evaluable_view(stencil, self.arr, offset)

    def evaluable_interpolation_view(self, stencil):
        return self._evaluable_view(stencil, self.interpolate_out)

    def evaluable_restriction_view(self, stencil):
        return self._evaluable_view(stencil, self.restrict_out)

    def compute_residual(self, stencil):
        if self.modified_rhs is False:
            self.res_mid[:] = self.rhs - stencil.eval_convolve(self.evaluable_view(stencil))
        else:
            # not sure if this works
            self.res_mid[:] = self.rhs - stencil.eval_convolve(self.mid, "same")

    def border_function_generator(self, stencil):
        """Generates a function which returns true if the index of the
           evaluable view is on the border, attention just works if evaluable view was generated!

        """

        def is_on_border(indice):
            in_the_middle = True

            for i in range(self.dim):
                in_the_middle = in_the_middle and \
                               (indice[i] >= stencil.b[i][0] and
                                indice[i] < (self.mid.shape[i] + stencil.b[i][1]))
            return not in_the_middle

        return is_on_border
