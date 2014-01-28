# coding=utf-8
"""
.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""


class ISolutionData(object):
    """
    Summary
    -------
    General storage for solution data.

    Notes
    -----
    It should not be necessary to directly and explicitly create instances of this and derived classes.
    """

    def __init__(self, *args, **kwargs):
        self._data = None
        self._dim = 0
        self._numeric_type = None

    @property
    def dim(self):
        """
        Summary
        -------
        Read-only accessor for the spacial dimension.

        Returns
        -------
        dim : :py:class:`int`
        """
        return self._dim

    @property
    def numeric_type(self):
        """
        Summary
        -------
        Read-only accessor for the numerical type.

        Returns
        -------
        numeric_type : :py:class:`numpy.dtype`
        """
        return self._numeric_type


__all__ = ['ISolutionData']
