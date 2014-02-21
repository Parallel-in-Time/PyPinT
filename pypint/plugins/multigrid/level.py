# coding=utf-8
# each dimension should have its own level class in order to
# simplify programming.

import numpy as np
import scipy.signal as sig
from pypint.utilities import assert_is_callable, assert_is_instance, \
                                     assert_condition
from pypint.plugins.multigrid.multigridproblem import MultiGridProblem
from pypint.plugins.multigrid.stencil import Stencil


class Level1D(np.ndarray):
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
    def __new__(cls, shape, mg_problem=None, max_borders=None,
                dtype=float, buffer=None, offset=0,
                strides=None, order=None):
        """
        Summary
        -------
        takes the physical MultiGridProblem and initialises with help of
        max_borders, and n_points an appropriate array
        """
        if isinstance(shape, int):
            if isinstance(max_borders, np.ndarray) and max_borders.size >= 2:
                forward_shape = shape + max_borders[0] + max_borders[1]
            elif max_borders is None:
                max_borders = np.asarray([0, 0])
                forward_shape = shape
            else:
                raise ValueError("Please provide an ndarray with the size of 2")
            obj = np.ndarray.__new__(cls, forward_shape, dtype, buffer, offset,
                                     strides, order)
            if isinstance(mg_problem, MultiGridProblem) \
                    and mg_problem.dimension == 1:
                obj._mg_problem = mg_problem
            else:
                raise ValueError("Please provide a MultiGridProblem")
        elif isinstance(shape, Level1D):
            # now we have a fallback solution if the borders are chosen wrong
            forward_shape = shape.size - shape.borders[0] - shape.borders[1]
            if isinstance(max_borders, np.ndarray) and max_borders.size >= 2:
                forward_shape = forward_shape + max_borders[0] + max_borders[1]
            else:
                max_borders = shape.borders
                forward_shape = forward_shape + max_borders[0] + max_borders[1]

            obj = np.ndarray.__new__(cls, forward_shape, shape.dtype,
                                     buffer, offset,
                                     shape.strides, order)

            obj[max_borders[0]:-max_borders[1]] = shape.mid

            if isinstance(mg_problem, MultiGridProblem) \
                    and mg_problem.dimension == 1:
                obj._mg_problem = mg_problem
            else:
                obj._mg_problem = shape.mg_problem
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
            obj = np.ndarray.__new__(cls, forward_shape, shape.dtype,
                                     buffer, offset,
                                     shape.strides, order)
            obj[max_borders[0]:-max_borders[1]] = shape

            if isinstance(mg_problem, MultiGridProblem) \
                    and mg_problem.dimension == 1:
                obj._mg_problem = mg_problem
            else:
                raise ValueError("Please provide a MultiGridProblem")


        else:
            raise TypeError("shape is in no shape")

        obj.borders = max_borders
        # gives view to the padded regions and the middle
        obj.left = obj[:obj.borders[0]]
        obj.right = obj[-obj.borders[1]:]
        obj.mid = obj[obj.borders[0]:-obj.borders[1]]
        return obj

    def __array_finalize__(self, obj):
        """
        Summary
        -------
        This function is called, everytime a new ndarray is constructed by
        one of the  following methods
            1. direct constructor call
            2. view casting , e.g. slicing
            3. or a creation from template
        """

        if obj is None:
            return
        # attributes to be inherited

        self.borders = getattr(obj, 'borders', None)
        self._mg_problem = getattr(obj, '_mg_problem', None)
        self.left = getattr(obj, 'left', None)
        self.right = getattr(obj, 'right', None)
        self.mid = getattr(obj, 'mid', None)

    def __array_prepare__(self, in_arr, context=None):
        """
        Summary
        -------
        called before a ufunc
        """

        return np.ndarray.__array_prepare__(self, in_arr[:], context)

    def __array_wrap__(self, out_arr, context=None):
        """
        Summary
        -------
        called after a ufunc
        """
        self.embed(out_arr)
        return np.ndarray.__array_wrap__(self, self.flatten(), context)

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
            l_f_b = np.linspace(-self.borders[0], -1, self.borders[0]) *\
                self._mg_problem.act_grid_distances[0] +\
                self._mg_problem.geometry[0][0]
            # right_from_border
            r_f_b = np.linspace(1, self.borders[1], self.borders[1]) *\
                self._mg_problem.act_grid_distances[0] +\
                self._mg_problem.geometry[0][1]
            #  left side
            self.left[:] = fl(l_f_b)
            #  right side
            self.right[:] = fr(r_f_b)


stencil = Stencil(3)
stencil[:] = np.asarray([1, -2, 1])
mg_prob = MultiGridProblem(stencil, lambda x: 5.)
lvl = Level1D(5, mg_prob, np.asarray([3, 3]))
print(type(lvl))
print(lvl)
# fuellen von werten
lvl.mid[:] = np.arange(5)
print(lvl)
lvl.pad()
print(lvl)
print(lvl.left)
print(lvl.mid)
print(lvl.right)
print(isinstance(lvl, Level1D))
lvl2 = Level1D(lvl, None, np.asarray([1, 1]))
lvl2.pad()
print(lvl2)
#print(lvl2)

#lvl2 = Level1D(lvl)
#print(lvl2)

#lvl.embed(np.arange(5))

