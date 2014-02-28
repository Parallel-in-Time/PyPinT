# coding=utf-8
import numpy as np
import scipy.signal as sig
from pypint.utilities import assert_is_callable, assert_is_instance, assert_condition

#Todo: Enhance the stencil class with useful methods,
#       which emerge as necessity in the development process

# diese klasse währe super, aber einfach nicht praktikabel in der Zeit die ich habe


class Stencil(np.ndarray):
    """
    Summary
    -------
    a class which knows its middle and borders
    """
    def __init__(self, arr, center=None):
        assert_is_instance(arr, np.ndarray, "the array is not a numpy array")
        if center is None:
            center = np.floor(np.asarray(arr.shape)*0.5)

        assert_is_instance(center, np.ndarray, "the center is not a np array")
        assert_condition(arr.ndim == center.size, ValueError,
                         "center does not match with stencil array")
        self.arr = arr
        self.dim = arr.ndim
        self.center = center
        # compute borders
        self.b = [[0.0, 0.0]]*self._dimension
        for i in range(self.dim):
            left = arr.shape[i] - self.center[i] - 1
            right = arr.shape[i] - left - 1
            self.b[self._dimension - i - 1] = [left, right]

    def eval(self, array_in, array_out):
        """Evaluate via scipy.signal.convolve

        """
        array_out[:] = sig.convolve(array_in, self.arr, 'valid')


class InterpolationStencil1D(object):
    """1D Stencil

    """

    def __init__(self, stencil_list, center=None):
        """init

        """
        # in the case of just one stencil
        if isinstance(stencil_list, np.ndarray) and arr_list.ndim == 1:
            self.mode = "own"
            self.increase_of_points = 2
            self.stencil_list = []
            self.stencil_list.append(stencil_list)
            self.center = []
            if center is None:
                self.center.append(np.floor(np.asarray(self.stencil[0].shape)*0.5))
            elif isinstance(center, int) and center < self.stencil[0].size:
                self.center.append(center)
            else:
                raise ValueError("Wrong argument")
        # now the case of a stencil set
        elif isinstance(stencil_list, list):
            # check if each one is a nd.array
            for stencil in stencil_list:
                assert_is_instance(stencil, Stencil,
                                   "not real stencil", self)
            # each stencil
                self.mode = "list"
                self.stencil_list = stencil_list
                self.increase_of_points = len(stencil_list)
        if self.mode == "own":
            self.eval = self.eval_own
        elif self.mode == "list":
            self.eval = self.eval_list
        else:
            raise RuntimeError("Something went terribly wrong")

    def eval_own(self, arrays_in, arrays_out):
        """Evaluation function if just one stencil is given

        There is no test if the in and out array are matching this is the task
        of the multilevel provider.
        """
        arrays_out[0][::2] = arrays_in
        arrays_out[0][1::2] = sig.convolve(arrays_out[0], self.stencil[0])[::2]

    def eval_list(self, arrays_in, arrays_out):
        """Evaluation function if more than one stencil is given

        There is no test if the in and out array are matching this is the task
        of the multilevel provider.
        """
        j = 0
        for stencil in self.stencil_list:
            arrays_out[j][j::self.increase_of_points] = \
                sig.convolve(arrays_in[j], stencil, 'valid')
            j = j+1


class RestrictionStencil(object):
    """Restriction stencil class

    """
    def __init__(self, restriction_stencil, decrease_in_points=None):
        assert_is_instance(restriction_stencil, Stencil, "Not an stencil")
        self.rst_stencil = restriction_stencil
        if decrease_in_points is None:
            self.dip = np.asarray(self.rst_stencil.arr.shape) - 1
        elif isinstance(decrease_in_points, np.ndarray) and\
                        decrease_in_points.ndim == self.rst_stencil.ndim:
            self.dip = decrease_in_points
        else:
            raise ValueError("Wrong decrease in points")
        self.dim = restriction_stencil.dim

        if self.dim == 1:
            self.eval = self.evalA_1D
        elif self.dim == 2:
            self.eval = self.evalA_2D
        elif self.dim == 3:
            self.eval = self.evalA_2D
        else:
            raise NotImplementedError("More than 3 dimensions " +
                                      "are not implemented")

    def evalA_1D(self, array_in, array_out):
        array_out = sig.convolve(array_in, self.rst_stencil)[::self.dip[0]]

    def evalA_2D(self, array_in, array_out):
        array_out = \
            sig.convolve(array_in, self.rst_stencil)[::self.dip[0], ::self.dip[1]]

    def evalA_3D(self, array_in, array_out):
        array_out = \
            sig.convolve(array_in, self.rst_stencil)[::self.dip[0], ::self.dip[1], ::self.dip[2]]

# TODO: a more effective evaluation strategy is needed
