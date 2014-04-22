# coding=utf-8

import numpy as np
import scipy.signal as sig
from pypint.plugins.multigrid.i_interpolation import IInterpolation
from pypint.plugins.multigrid.stencil import Stencil
from pypint.utilities import assert_is_callable, assert_is_instance, assert_condition
from pypint.plugins.multigrid.i_multigrid_level import IMultigridLevel
import itertools as it

class InterpolationByStencilForLevels(IInterpolation):
    """1D class for Interpolation which binds two levels
        This Interpolationclass implicitly assumes the following structure
       B_ _,_ _,_ _,_ _,_ _,_ _,_ _B    <- Level in
         B_._._._._._._._._._._._B     <- Level out

        It also checks if under this assumption the interpolation is possible.

        The format of stencil matrix is the following
        stencil_matrix= [ (Stencil 1, Position 1),(Stencil 2, Position 2), . . .]
    """
    def __init__(self, stencil_list, level_in, level_out, *args, **kwargs):
        """init
        """
        # check if all parameters are fitting
        assert_is_instance(stencil_list, list)

        for st in stencil_list:
            assert_is_instance(st[0], Stencil, "that is not a stencil")
        self.stencil_list = stencil_list
        assert_is_instance(level_in, IMultigridLevel, "Not a IMultigridLevel")
        assert_is_instance(level_out, IMultigridLevel, "Not a IMultigridLevel")
        self.level_in = level_in
        self.level_out = level_out
        # increase in points for each direction
        self.iip = []
        for i in range(level_in.mid.ndim):
            self.iip.append((level_out.mid.shape[i]-1)/(level_in.mid.shape[i]-1) - 1)
            # print("in.shape[", i, "]: ", level_in.mid.shape[i])
            # print("out.shape[", i, "]: ", level_out.mid.shape[i])
            # print("iip[", i, "]: ", self.iip[-1])
            if (self.iip[-1] % 1) != 0:
                raise ValueError("The Levels do not match in direction " + str(i))
        # compute evaluable views with the positions
        self.evaluable_views = []
        self.slices = []
        for st, pos in stencil_list:
            sl = []
            for i in range(st.dim):
                sl.append(slice(pos[i], None, self.iip[i]))
            self.evaluable_views.append(level_in.evaluable_view(st))
            self.slices.append(tuple(sl.copy()))

        super().__init__(*args, **kwargs)

    def eval(self):
        """ for each stencil at a certain position the convolution is computed

        """
        for i in range(len(self.stencil_list)):
            self.level_out.mid[self.slices[i]] = \
                sig.convolve(self.evaluable_views[i], self.stencil_list[i][0], 'valid')


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
