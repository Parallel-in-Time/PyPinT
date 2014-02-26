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

    def eval_own(self, array_in, array_out):
        """Evaluation function if just one stencil is given

        There is no test if the in and out array are matching this is the task
        of the multilevel provider.
        """
        array_out[::2] = array_in
        array_out[1::2] = sig.convolve(array_out, self.stencil[0])[::2]

    def eval_list(self, array_in, array_out):
        """Evaluation function if more than one stencil is given

        There is no test if the in and out array are matching this is the task
        of the multilevel provider.
        """
        j = 0
        for stencil in self.stencil_list:
            array_out[j::self.increase_of_points] = \
                sig.convolve(array_in, stencil, 'valid')
            j = j+1
