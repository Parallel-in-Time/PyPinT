# coding=utf-8
# each dimension should have its own level class in order to
# simplify programming.

import numpy as np
import scipy.signal as sig

from pypint.utilities import assert_is_callable, assert_is_instance, \
                                     assert_condition
from pypint.plugins.multigrid.multigrid_problem import MultiGridProblem
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

    Examples
    --------

    """
    def __init__(self, shape, mg_problem=None, max_borders=None,
                dtype=float):
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
            if isinstance(mg_problem, MultiGridProblem) \
                    and mg_problem.dimension == 1:
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

            if isinstance(mg_problem, MultiGridProblem) \
                    and mg_problem.dimension == 1:
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

            if isinstance(mg_problem, MultiGridProblem) \
                    and mg_problem.dimension == 1:
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
    # def __init__(self, obj):
    #     self = np.ndarray(obj)
    #     return self

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
            fl = self._mg_problem.boundary_functions[0][0]
            fr = self._mg_problem.boundary_functions[0][1]
                # left from border
            l_f_b = self.space_tensor[0:self.borders[0]]
            # right_from_border
            r_f_b = self.space_tensor[-self.borders[1]:]
            #  left side
            self.left[:] = fl(l_f_b)
            #  right side
            self.right[:] = fr(r_f_b)

    def evaluable_view(self, stencil, offset = 0):
        """gives the right view of the array

        """
        if self.dim == 1:
            if isinstance(stencil, Stencil):

                l = self.borders[0]-stencil.b[0][0]
                r = -(self.borders[1]-stencil.b[0][1])
            else:
                l = self.borders[0]-stencil[0][0]
                r = -(self.borders[1]-stencil[0][1])
            return self.arr[l+offset: r+offset]
        else:
            raise NotImplementedError("Another dimension than one "
                                      "is not supplied")

    def border_function_generator(self, stencil):
        """Generates a function which returns true if the index of the
           evaluable view is on the border, attention just works if evaluable view was generated!

        """

        def is_on_border(indice):
            for i in range(self.dim):
                if indice[0] < stencil.b[0][0] or indice[0] >= self.mid.shape[0]+stencil.b[0][0]:
                    return True
        return is_on_border

    # def __copy__(self):
    #     copy = self.__class__.__new__(self.__class__)
    #     copy.__dict__.update(self.__dict__)
    #     return copy
    #
    # def __deepcopy__(self, memo):
    #     copy = self.__class__.__new__(self.__class__)
    #     memo[id(self)] = copy
    #     for item, value in self.__dict__.items():
    #         setattr(copy, item, deepcopy(value, memo))
    #     return copy


# stencil = Stencil(3)
# stencil[:] = np.asarray([1, -2, 1])
# mg_prob = MultiGridProblem(stencil, lambda x: 5.)
# lvl = MultigridLevel1D(5, mg_prob, np.asarray([3, 3]))
# print(type(lvl))
# print(lvl)
# # fuellen von werten
# lvl.mid[:] = np.arange(5)
# print(lvl)
# lvl.pad()
# print(lvl)
# print(lvl.left)
# print(lvl.mid)
# print(lvl.right)
# print(isinstance(lvl, MultigridLevel1D))
#lvl2 = MultigridLevel1D(lvl, None, np.asarray([1, 1]))
# a = lvl[1:3]
# print(type(a))
# print(a.left)
# print(a.right)
# print(a.mid)
# print(lvl[1:2])
# a = lvl[1:3]
# print(type(a))
# print(a.__array__())
# lvl2.pad()
# print(lvl2)
# print(type(lvl.mid))
#print(lvl2)

#lvl2 = MultigridLevel1D(lvl)
#print(lvl2)

#lvl.embed(np.arange(5))

