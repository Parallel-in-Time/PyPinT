# coding=utf-8

import numpy as np
from pypint.multi_level_providers.multi_level_provider import MultiLevelProvider
from pypint.utilities import assert_is_callable, assert_is_instance, assert_condition
from pypint.plugins.multigrid.stencil import Stencil, InterpolationStencil1D, RestrictionStencil
from pypint.plugins.multigrid.level import MultiGridLevel1D
import scipy.signal as sig

class Smoother(object):
    """Smoother Root Class for Multigrid

    """

    def __init__(self, dimension=1, **kwds):
        self.dim = dimension
        if kwds["smoothing_function"] is None:
            def smoothing_function():
                assert not hasattr(super(), 'smoothing_function')
            self.smoothing_function = smoothing_function
        else:
            self.smoothing_function = kwds["smoothing_function"]



class SplitSmoother(Smoother):
    """ A general Smoothing class which arises from splitting the main stencil

    This class of smoothers is easy to derive and really broad,
    it is statically linked to a certain level.
    """

    def __init__(self, l_plus, l_minus, level, **kwargs):
        """init method of the split smoother

        l_plus and l_minus have to be centralized
        """
        assert_is_instance(l_plus, np.ndarray, "L plus has to be a np array")
        assert_is_instance(l_minus, np.ndarray, "L minus has to be a np array")
        assert_condition(l_plus.shape == l_minus.shape, TypeError,
                         "It is not an splitting")
        # check the level !has to be improved! inheritance must exist for
        # level class
        assert_is_instance(level, MultiGridLevel1D)

        self.lvl = level
        self.l_plus = l_plus
        self.l_minus = l_minus
        self.st_minus = Stencil(l_minus)
        self.lvl_view_inner = level.evaluable_view(self.st_minus)
        self.lvl_view_outer = \
            level.evaluable_view(np.asarray(self.st_minus.arr.shape)*2)

        grid = self.lvl_view_inner.shape
        self.st_plus = Stencil(l_plus, grid=grid, solver="factorize")

        # construct stencils

        super().__init__(l_plus.ndim)

    def relax(self, n=1):
        """Does the relaxation step several times on the lvl

        the hardship in this case is to use the ghostcells accordingly
        because one has a two operators which are applied,
        and this results in a broader border of ghostcells.
        Parameters
        ----------
            n : Integer
                relax n times
        """

        for i in range(n):
            self.lvl_view_inner = \
                self.l_plus.solver(
                    -self.l_minus.eval_convolve(
                        self.lvl_view_outer).reshape(-1)).reshape(
                            self.lvl_view_inner.shape)

            self.lvl.pad()
