# coding=utf-8
# each dimension should have its own level class in order to
# simplify programming.

import numpy as np
import scipy.signal as sig
from pypint.utilities import assert_is_callable, assert_is_instance, assert_condition
from pypint.plugins.multigrid.multigrid_problem_mixin import problem_is_multigrid_problem
from pypint.plugins.multigrid.stencil import Stencil
from pypint.plugins.multigrid.i_multigrid_level import IMultigridLevel


class MultigridLevel1D(IMultigridLevel):
    """
    Summary
    -------
    Simple extension of an numpy array, which allows
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
    def __init__(self, shape, mg_problem=None, max_borders=None, dtype=float, role="ML"):
        """
        Summary
        -------
        takes the physical MultiGridProblem and initialises with help of
        max_borders, and n_points an appropriate array
        """
        # the level should know its geometrical information, because it differs from level to level
        self.space_tensor = None

        if isinstance(shape, int):
            if isinstance(max_borders, np.ndarray) and max_borders.size >= 2:
                forward_shape = shape + max_borders[0] + max_borders[1]
            elif max_borders is None:
                max_borders = np.asarray([0, 0])
                forward_shape = shape
            else:
                raise ValueError("Please provide an ndarray with the size of 2")
            self.arr = np.zeros(forward_shape, dtype=dtype)
            if problem_is_multigrid_problem(mg_problem, checking_obj=self) and len(mg_problem.spacial_dim) == 1:
                self._mg_problem = mg_problem
            else:
                raise ValueError("Please provide a MultiGridProblem")
        elif isinstance(shape, MultigridLevel1D):
            # now we have a fallback solution if the borders are chosen wrong
            forward_shape = shape.size - shape.borders[0] - shape.borders[1]
            if isinstance(max_borders, np.ndarray) and max_borders.size >= 2:
                forward_shape = forward_shape + max_borders[0] + max_borders[1]
            else:
                max_borders = shape.borders
                forward_shape = forward_shape + max_borders[0] + max_borders[1]

            self.arr = np.zeros(forward_shape, dtype=dtype)

            self.arr[max_borders[0]:-max_borders[1]] = shape.mid

            if problem_is_multigrid_problem(mg_problem, checking_obj=self) and mg_problem.dim == 1:
                self._mg_problem = mg_problem
            else:
                self._mg_problem = shape.mg_problem
        elif isinstance(shape, np.ndarray):
            # in this case new memory has to be wasted because we have to
            # embed this array into a bigger one
            # but we at least use all properties of the template array
            shape = shape.flatten()
            if isinstance(max_borders, np.ndarray) and max_borders.size >= 2:
                forward_shape = shape.size + max_borders[0] + max_borders[1]
            elif max_borders is None:
                max_borders = np.asarray([0, 0])
                forward_shape = shape.size
            else:
                raise ValueError("Please provide an ndarray with the size of 2")

            self.arr = np.zeros(forward_shape, dtype=dtype)
            self.arr[max_borders[0]:-max_borders[1]] = shape

            if problem_is_multigrid_problem(mg_problem, checking_obj=self) and mg_problem.dim == 1:
                self._mg_problem = mg_problem
            else:
                raise ValueError("Please provide a MultiGridProblem")

        else:
            raise TypeError("shape is in no shape")

        self.borders = max_borders
        # gives view to the padded regions and the middle
        # here it is important to use __array__, because
        # the different parts are just ndarrays and not another MultigridLevel1D objects
        self.left = self.arr.__array__()[:self.borders[0]]
        self.right = self.arr.__array__()[-self.borders[1]:]
        self.mid = self.arr.__array__()[self.borders[0]:-self.borders[1]]
        self.rhs = np.copy(self.mid)

        # the first border points coincides with the geometrical border
        # that is why self.mid.size+1 is used instead of self.mid.size - 1
        self.h = (self._mg_problem.geometry[0][1]
                  - self._mg_problem.geometry[0][0]) / (self.mid.size + 1)
        start = self._mg_problem.geometry[0][0] - self.h * (max_borders[0] - 1)
        stop = self._mg_problem.geometry[0][1] + self.h * (max_borders[1] - 1)
        self.space_tensor = np.linspace(start, stop, self.arr.size)
        self._mid_points = self.mid.size
        self.dim = 1

        # set the interpolation and restriction ports according to the level which is used
        self.role = role
        # some place to store the residuum
        self.res = np.zeros(self.arr.shape)
        # some views on the residuum
        self.res_left = self.res.__array__()[:self.borders[0]]
        self.res_right = self.res.__array__()[-self.borders[1]:]
        self.res_mid = self.res.__array__()[self.borders[0]:-self.borders[1]]

        # it would be nicer to have the slices of the different parts
        self.mid_slice = slice(self.borders[0], self.borders[1])
        self.left_slice = slice(None, self.borders[0])
        self.right_slice = slice(-self.borders[1], None)

        if role is "FL":
            # here we define the ports for the finest level
            self.interpolate_out = None
            self.interpolate_in = self.mid
            self.restrict_in = None
            self.restrict_out = self.res
            self.restriction_out_mid = self.res_mid
            # adjus boundary functions
            self.fl = self._mg_problem.boundary_functions[0][0]
            self.fr = self._mg_problem.boundary_functions[0][1]
        elif role is "ML":
            # here we define the ports for the mid level
            self.interpolate_out = self.arr
            self.interpolate_out_mid = self.mid
            self.interpolate_in = self.mid
            self.restrict_in = self.rhs
            self.restrict_out = self.res
            self.restriction_out_mid = self.res_mid
            self.fl = lambda x: 0.
            self.fr = lambda x: 0.
        elif role is "CL":
            # here we define the ports for the coarsest level
            self.interpolate_out = self.arr
            self.interpolate_out_mid = self.mid
            self.interpolate_in = None
            self.restrict_in = self.rhs
            self.restrict_out = None
            self.fl = lambda x: 0.
            self.fr = lambda x: 0.
        else:
            raise ValueError("MultiLevel has no role "+self.role)

        # in order to know if the rhs was modified
        self.modified_rhs = False

    def adjust_references(self):
        self.left = self.arr.__array__()[:self.borders[0]]
        self.right = self.arr.__array__()[-self.borders[1]:]
        self.mid = self.arr.__array__()[self.borders[0]:-self.borders[1]]


    @property
    def mg_problem(self):
        """
        return MultiGridProblem
        """
        return self._mg_problem

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
        assert_condition(ue.size == self._mid_points, ValueError,
                         "Array to embed has the wrong size")
        self[self.borders[0]:-self.borders[1]] = ue

    def pad(self):
        """
        Summary
        -------
        Uses the informations in Multigridproblems in order to
        pad the array.
        """
        if self._mg_problem.boundaries[0] == 'periodic':
            #  left side
            self.left[:] = self.mid[-self.borders[0]:]
            #  right side
            self.right[:] = self.mid[:self.borders[1]]
        elif self._mg_problem.boundaries[0] == 'dirichlet':

                # left from border
            l_f_b = self.space_tensor[0:self.borders[0]]
            # right_from_border
            r_f_b = self.space_tensor[-self.borders[1]:]
            #  left side
            self.left[:] = self.fl(l_f_b)
            #  right side
            self.right[:] = self.fr(r_f_b)

    def _evaluable_view(self, stencil, arr, offset=0):
        """gives the right view of the array

        """
        if self.dim == 1:
            if isinstance(stencil, Stencil):

                l = self.borders[0]-stencil.b[0][0]
                r = -(self.borders[1]-stencil.b[0][1])
            else:
                l = self.borders[0]-stencil[0][0]
                r = -(self.borders[1]-stencil[0][1])
            return arr[l+offset: r+offset]
        else:
            raise NotImplementedError("Another dimension than one "
                                      "is not supplied")

    def evaluable_view(self, stencil, offset=0):
        """gives the right view of the array

        """
        return self._evaluable_view(stencil, self.arr, offset)

    def evaluable_interpolation_view(self, stencil):
        return self._evaluable_view(stencil, self.interpolate_out)

    def evaluable_restriction_view(self, stencil):
        return self._evaluable_view(stencil, self.restrict_out)

    def compute_residual(self, stencil):
        if self.modified_rhs is False:
            self.res_mid[:] = self.rhs - stencil.eval_convolve(self.evaluable_view(stencil)) / self.h**2
        else:
            self.res_mid[:] = self.rhs - stencil.eval_convolve(self.mid, "same") / self.h**2


    def border_function_generator(self, stencil):
        """Generates a function which returns true if the index of the
           evaluable view is on the border, attention just works if evaluable view was generated!

        """

        def is_on_border(indice):
            for i in range(self.dim):
                if indice[0] < stencil.b[0][0] or indice[0] >= self.mid.shape[0]+stencil.b[0][0]:
                    return True
        return is_on_border



