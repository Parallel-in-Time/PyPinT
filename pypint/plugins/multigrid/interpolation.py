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
    def __init__(self, stencil_list, level_in, level_out, *args, pre_assign=None, **kwargs):
        """init
        """
        super(InterpolationByStencilForLevels, self).__init__(*args, **kwargs)

        if pre_assign is None:
            # no pre assignment function
            self.pre_assign = lambda a, b: b
        else:
            assert_is_callable(pre_assign, "Pre assignment function is not callable")
            self.pre_assign = pre_assign

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
        # compute evaluable views with the positions and slices
        self.evaluable_views = []
        self.slices_out = []
        self.slices_in = []
        for st, pos in stencil_list:
            sl_out = []
            sl_in = []
            for i in range(st.dim):
                sl_out.append(slice(pos[i], None, self.iip[i]+1))
                if pos[i] == 0:
                    sl_in.append(slice(None, None))
                else:
                    sl_in.append(slice(0, -1))
            # print("Stencilcenter : ", st.center)
            # print("Stencilborder :", st.b)
            self.evaluable_views.append(level_in.evaluable_interpolation_view(st))
            # print("The view: \n", self.evaluable_views[-1])
            self.slices_out.append(tuple(sl_out.copy()))
            self.slices_in.append(tuple(sl_in.copy()))



    def eval(self):
        """ for each stencil at a certain position the convolution is computed

        """
        for i in range(len(self.stencil_list)):
            # sigs = sig.convolve(self.evaluable_views[i], self.stencil_list[i][0].arr, 'valid')
            print(i, self.level_out.interpolate_in[self.slices_out[i]].shape, self.evaluable_views[i].shape, self.stencil_list[i][0].arr[::-1].shape)
            # print("slice_out :\n", self.slices_out[i])
            # print("Stencil_arr: \n", self.stencil_list[i][0].arr)
            # print("eval_view: \n", self.evaluable_views[i])
            # print("sig: \n", sig.convolve(self.evaluable_views[i], self.stencil_list[i][0].arr[::-1], 'valid')[self.slices_in[i]])

            # here i learned that the convolution has to reverse the stencil array in order to work like a stencil

            self.level_out.interpolate_in[self.slices_out[i]] = \
                self.pre_assign(
                    self.level_out.interpolate_in[self.slices_out[i]],
                    sig.convolve(self.evaluable_views[i],
                                 self.stencil_list[i][0].arr[::-1], 'valid')[self.slices_in[i]])


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
        arrays_out[0][1::2] = sig.convolve(arrays_out[0], self.stencil[0][::-1])[::2]

    def eval_list(self, arrays_in, arrays_out):
        """Evaluation function if more than one stencil is given

        There is no test if the in and out array are matching this is the task
        of the multilevel provider.
        """
        j = 0
        for stencil in self.stencil_list:
            arrays_out[j][j::self.increase_of_points] = \
                sig.convolve(arrays_in[j], stencil[::-1], 'valid')
            j = j+1
