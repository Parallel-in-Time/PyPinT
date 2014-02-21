# coding=utf-8
import numpy as np


#Todo: Enhance the stencil class with useful methods,
#       which emerge as necessity in the development process


class Stencil(np.ndarray):
    """
    Summary
    -------
    a np.ndarray which knows its middl
    """
    def __new__(cls, shape, stencil_center=None,
                dtype=float, buffer=None, offset=0,
                strides=None, order=None):
        obj = np.ndarray.__new__(cls, shape, dtype, buffer, offset, strides, order)

        if stencil_center is None:
            obj.center = np.floor(np.asarray(shape)*0.5)
        else:
            obj.center = stencil_center
        return obj

    def __array_finalize__(self, obj):
        """finalize the construction of the stencil

        """

        if obj is None:
            return

        self.center = getattr(obj, 'borders', None)
