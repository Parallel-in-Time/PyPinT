# coding=utf-8
class IMultigridSmoother(object):
    """IMultigridSmoother Root Class for Multigrid

    """

    def __init__(self, dimension=1, *args , **kwds):
        self.dim = dimension
        if "smoothing_function" not in kwds:
            def smoothing_function():
                assert not hasattr(super(), 'smoothing_function')
            self.smoothing_function = None
        else:
            self.smoothing_function = kwds["smoothing_function"]
