# coding=utf-8

import numpy as np
import scipy.signal as sig
from pypint.plugins.multigrid.i_interpolation import IInterpolation
from pypint.plugins.multigrid.stencil import Stencil
from pypint.utilities import assert_is_instance



class InterpolationByStencilListIn1D(IInterpolation):
    """1D Class for Interpolation

    """

    def __init__(self, stencil_list, center=None, *args, **kwargs):
        """init

        """
        # in the case of just one stencil
        if isinstance(stencil_list, np.ndarray) and stencil_list.ndim == 1:
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
        super().__init__(*args, **kwargs)

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
