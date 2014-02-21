# coding=utf-8
"""
MultigridLevelProvider
"""
import numpy as np
from pypint.multi_level_providers.multi_level_provider import MultiLevelProvider
from pypint.utilities.tracing import assert_is_callable, assert_is_instance, assert_condition
import scipy.signal as sig

class InterpolationOperator

class MultiGridLevelProvider(object):
    """
    Summary
    _______
    Contains all interpolation and restriction operators.
    """
    def __init__(self, number_of_levels, interpolation_stencil_set, restriction_stencil_set):
        self._nol = number_of_levels
        assert_is_instance(interpolation_stencil_set, list)
        assert_is_instance(restriction_stencil_set, list)
        nipl = len(interpolation_stencil_set)
        nrst = len(restriction_stencil_set)
        self._nipl = nipl
        self._nrst = nrst
        assert_condition((self._nol == nipl or nipl == 1)
                         and (self._nol == nrst or nrst == 1),
                         "The number of interpolation and restriction operators"
                         + " should be 1 or the number of levels")
        self._ipls = interpolation_stencil_set
        self._rsts = restriction_stencil_set

        self._ipl_mult = []
        self._rst_divi = []

        for obj in restriction_stencil_set:
            assert_is_instance(obj, np.ndarray, "One of the restriction"
                                                + "stencils is not an np.array")
            self._rst_divi.append(obj.shape[0])
        for obj in interpolation_stencil_set:
            assert_is_instance(obj, np.ndarray, "One of the interpolation"
                                                + "stencils is not an np.array")
        # es wird vermutlich eine eigene definition geben.
        # fuer die interpolation finden wir nun heraus wie viele punkte hinzu kommen
            self._ipl_mult.append(obj.shape[0])

        # es sollte ueberlegt werden welches level gerade behandelt wird
        # hiermit entscheide ich vorerst, dass 0 das feinste level ist


        self._akt_level = 0

    def set_akt_level(self, level):
        """
        returns the current level
        Parameters
        ----------
        level : integer
            indicates on which level the interpolation is done
        """
        assert_condition(level % 1 == 0 and level < self._nol,
                         "level is too big or a float")
        self._akt_level = level

    def akt_ipl(self, level=None):
        """
        returns the current interpolation stencil
        Parameters
        ----------
        level : integer
            indicates on which level the interpolation is done
        """
        if level is not None:
            self.set_akt_level(level)
        if self._nipl == 1:
            return self._ipls[0]
        else:
            return self._ipls[self._akt_level]

    def interpolate(self, u_pad, level=None):
        """
        The main interpolation function,
        if the interpolation stencil is bigger than the system stencil
        the boundaries are computed wrong. Hence it is important to have
        the padding of u right.
        Parameters
        ----------
        u_pad : ndarray
            the padded value region
        level : integer
            indicates on which level the interpolation is done
        """
        if level is not None:
            self.set_akt_level(level)
        # anhand des multiplikationsfaktors wird die groesse der zureckgegebenen
        # matrix berrechnet
        dim = u_pad.ndim
        ipA = np.zeros(((np.asarray(u_pad.shape)-1)
                        * np.asarray(self._ipl_mult))+1)
        ipl = self.akt_ipl()

        if dim == 1:
            for i in range(self._ipl_mult[0]):
                ipA[i::self._ipl_mult[0]] = sig.convolve(u_pad, ipl[i], 'valid')
        elif dim == 2:
            for i in range(self._ipl_mult[0]):
                for j in range(self._ipl_mult[1]):
                    ipA[i::self._ipl_mult[0], j::self._ipl_mult[1]] = \
                        sig.convolve(u_pad, ipl[i, j], 'valid')
        else:
            print("Wer will den sowas!")



    def restrict(self, u_pad, level=None):
        """
        The main restriction operator.
        Parameters
        ----------
        u_pad : ndarray
            the padded value region
        level : integer
            indicates on which level the interpolation is done
        """
        if level is not None:
            self.set_akt_level(level)
        #
        dim = u_pad.ndim
        if dim == 1:
            repA = u_pad[]
