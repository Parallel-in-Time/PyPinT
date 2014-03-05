# coding=utf-8

import numpy as np
from pypint.multi_level_providers.multi_level_provider import MultiLevelProvider
from pypint.utilities import assert_is_callable, assert_is_instance, assert_condition
from pypint.plugins.multigrid.stencil import Stencil, InterpolationStencil1D, RestrictionStencil
from pypint.plugins.multigrid.level import Level1D
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

    This class of smoothers is
    """

    def __init__(self, l_plus, l_minus, **kwds):
        assert_is_instance(l_plus, np.ndarray, "L plus has to be a np array")
        assert_is_instance(l_minus, np.ndarray, "L minus has to be a np array")
        assert_condition(l_plus.shape == l_minus.shape, TypeError,
                         "It is not an splitting")
        self.l_plus = l_plus
        self.l_minus = l_minus

        super().__init__(l_plus.ndim)

    def relax(self, lvl, times=1):
        """Does the relaxation step several times on the lvl"""
        tmp = lvl
        for i in range(times):
            tmp = self.l_plus.solve(-self.l_minus.eval(tmp))

        return tmp
