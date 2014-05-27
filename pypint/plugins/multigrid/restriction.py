# coding=utf-8
import numpy as np
import scipy.signal as sig
# import scipy.sparse as sprs
# import scipy.sparse.linalg as spla
# import functools as ft
from pypint.utilities import assert_is_callable, assert_is_instance, assert_condition
from pypint.plugins.multigrid.i_multigrid_level import IMultigridLevel
from pypint.plugins.multigrid.stencil import Stencil
from pypint.plugins.multigrid.i_restriction import IRestriction

# TODO: Entwerfe eine Restrictions und Interpolationsklasse die auf FFT beruht, fuer den Vergleich
class RestrictionByStencilForLevelsClassical(IRestriction):
    """Restriction Stencil class which binds two level to each other, takes a
        Stencil object, and checks if they are compatible. If there is a possibility for restriction
        which is not standard it warns you about it.
        This Restrictionclass implicitly assumes the following structure
         B_._._._._._._._._._._._B     <- Level in
         B_ _,_ _,_ _,_ _,_ _,_ _B    <- Level out

    """
    def __init__(self, level_in, level_out, rst_stencil, *args, pre_assign=None, **kwargs):
        super(RestrictionByStencilForLevelsClassical, self).__init__(*args, **kwargs)
        if pre_assign is None:
            # no pre assignment function
            self.pre_assign = lambda a, b: b
        else:
            assert_is_callable(pre_assign, "Pre assignment function is not callable")
            self.pre_assign = pre_assign

        assert_is_instance(rst_stencil, Stencil, "Not a Stencil")
        assert_is_instance(level_in, IMultigridLevel, "Not a IMultigridLevel")
        assert_is_instance(level_out, IMultigridLevel, "Not a IMultigridLevel")
        self.level_in = level_in
        self.level_out = level_out
        self.rst_stencil = rst_stencil
        # check if the number of points per level matches
        self.dip = []
        for i in range(rst_stencil.dim):
            self.dip.append((level_in.mid.shape[i]-1)/(level_out.mid.shape[i]) - 1)
            print("in.shape[", i, "]: ", level_in.mid.shape[i])
            print("out.shape[", i, "]: ", level_out.mid.shape[i])
            print("dip[", i, "]: ", self.dip[-1])
            if (self.dip[-1] % 1) != 0:
                raise ValueError("The Level do not match in direction " + str(i))

        # now just construct a slice tuple and the evaluable view from the finer grid
        self.slices = []
        self.reverse_slice = []
        for i in range(rst_stencil.dim):
            self.slices.append(slice(None, None, self.dip[i]+1))
            self.reverse_slice.append(slice(None, None, -1))

        self.reversed_stencil = rst_stencil.arr[self.reverse_slice]


    def restrict(self):
        """Uses an unefficient algorithm in order to compute the restriction,
           because the convolution is computed on each node of the fine grid instead on every second or third
        """
        
        self.level_out.restrict_in[:] = self.pre_assign(self.level_out.restrict_in[:],
                                                       sig.convolve(self.level_in.restriction_out_mid,
                                                                    self.reversed_stencil,
                                                                    "valid")[self.slices])

class RestrictionByStencilForLevels(IRestriction):
    """Restriction Stencil class which binds two level to each other, takes a
        Stencil object, and checks if they are compatible. If there is a possibility for restriction
        which is not standard it warns you about it.
        This Restrictionclass implicitly assumes the following structure
         B_._._._._._._._._._._._B     <- Level in
       B_ _,_ _,_ _,_ _,_ _,_ _,_ _B    <- Level out

    """
    def __init__(self, rst_stencil, level_in, level_out, *args, pre_assign=None, **kwargs):
        super(RestrictionByStencilForLevels, self).__init__(*args, **kwargs)
        if pre_assign is None:
            # no pre assignment function
            self.pre_assign = lambda a, b: b
        else:
            assert_is_callable(pre_assign, "Pre assignment function is not callable")
            self.pre_assign = pre_assign

        assert_is_instance(rst_stencil, Stencil, "Not a Stencil")
        assert_is_instance(level_in, IMultigridLevel, "Not a IMultigridLevel")
        assert_is_instance(level_out, IMultigridLevel, "Not a IMultigridLevel")
        self.l_in = level_in
        self.l_out = level_out
        self.rst_stencil = rst_stencil
        # check if the number of points per level matches
        self.dip = []
        for i in range(rst_stencil.dim):
            self.dip.append((level_in.mid.shape[i]-1)/(level_out.mid.shape[i]-1) - 1)
            # print("in.shape[", i, "]: ", level_in.mid.shape[i])
            # print("out.shape[", i, "]: ", level_out.mid.shape[i])
            # print("dip[", i, "]: ", self.dip[-1])
            if (self.dip[-1] % 1) != 0:
                raise ValueError("The Level do not match in direction " + str(i))

        # now just construct a slice tuple and the evaluable view from the finer grid
        self.evaluable_view = level_in.evaluable_restriction_view(rst_stencil)
        self.slices = []
        for i in range(rst_stencil.dim):
            self.slices.append(slice(None, None, self.dip[i]+1))


    def restrict(self):
        """Uses an unefficient algorithm in order to compute the restriction,
           because the convolution is computed on each node of the fine grid instead on every second or third
        """
        self.l_out.restrict_in[:] = self.pre_assign(self.l_out.restrict_in[:],
                                                       sig.convolve(self.evaluable_view,
                                                                    self.rst_stencil.arr[::-1],
                                                                    "valid")[self.slices])

class RestrictionStencilPure(IRestriction):
    """Restriction stencil class just for nd arrays

    """
    def __init__(self, restriction_stencil, decrease_in_points=None, center=None):
        assert_is_instance(restriction_stencil, np.ndarray, "Not an ndarray")
        self.rst_stencil = Stencil(restriction_stencil, center)


        if decrease_in_points is None:
            self.dip = np.asarray(self.rst_stencil.shape) - 1
        elif isinstance(decrease_in_points, np.ndarray) and \
                        decrease_in_points.ndim == restriction_stencil.ndim:
            self.dip = decrease_in_points
        elif isinstance(decrease_in_points, int):
            self.dip = np.asarray([decrease_in_points]*restriction_stencil.ndim)
        else:
            raise ValueError("Wrong decrease in points")
        self.dim = restriction_stencil.ndim

        if self.dim == 1:
            self.eval = self.evalA_1D
        elif self.dim == 2:
            self.eval = self.evalA_2D
        elif self.dim == 3:
            self.eval = self.evalA_3D
        else:
            raise NotImplementedError("More than 3 dimensions " +
                                      "are not implemented")

    def evalA_1D(self, array_in, array_out):
        array_out[:] = sig.convolve(array_in, self.rst_stencil.arr[::-1], "valid")[::self.dip[0]]

    def evalA_2D(self, array_in, array_out):
        array_out[:] = \
            sig.convolve(array_in, self.rst_stencil.arr[::-1])[::self.dip[0], ::self.dip[1]]

    def evalA_3D(self, array_in, array_out):
        array_out[:] = \
            sig.convolve(array_in, self.rst_stencil.arr[::-1])[::self.dip[0], ::self.dip[1], ::self.dip[2]]

# TODO: a more effective evaluation strategy is needed
#       for example by writing a rst_stencil to sparse matrix function,
#       assumable the best way. And a great exercise
