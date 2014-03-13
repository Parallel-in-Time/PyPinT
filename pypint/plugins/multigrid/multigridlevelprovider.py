# coding=utf-8
"""
MultigridLevelProvider
"""
import numpy as np
from pypint.multi_level_providers.multi_level_provider import MultiLevelProvider
from pypint.utilities import assert_is_callable, assert_is_instance, assert_condition
from pypint.plugins.multigrid.stencil import Stencil, InterpolationStencil1D, RestrictionStencil
from pypint.plugins.multigrid.level import MultiGridLevel1D
from pypint.plugins.multigrid.multigrid_smoother import Smoother, SplitSmoother
import scipy.signal as sig


# uebernahme der Pypint konventionen, verwenden von LevelTransitionOperatoren

class StencilBasedLevelTransitionProvider1D(object):
    """Takes two stencils and constructs an instance

    with the according interpolation and restriction functions
    """
    def __init__(self, fine_level, coarse_level, rst_stencil, ipl_stencil):
        assert_is_instance(fine_level, MultiGridLevel1D, "Not an MultiGridLevel1D")
        assert_is_instance(coarse_level, MultiGridLevel1D, "Not an MultiGridLevel1D")

        self.fl = fine_level
        self.cl = coarse_level

        assert_is_instance(ipl_stencil, InterpolationStencil1D)
        assert_is_instance(rst_stencil, RestrictionStencil)
        assert_condition(rst_stencil.ndim == 1,
                         ValueError, "Restriction Stencil"
                         + "has not the dimension 1")
        self.ipl = ipl_stencil
        self.rst = rst_stencil
        self.ipl_fine_views = []
        self.ipl_coarse_views = []
        # collect the views which are needed,
        if self.ipl.mode == "own":
            self.ipl_fine_views.append(self.fl.evaluable_view(
                self.ipl.stencil_list[0]))
            self.ipl_coarse_views(self.cl.mid)
        elif self.ipl.mode == "list":
            for stencil in self.ipl.stencil_list:
                self.ipl_fine_views.append(self.fl.evaluable_view(stencil))
                self.ipl_coarse_views.append(self.cl.mid)
        else:
            raise NotImplementedError("What do you have in mind?")

        self.rst_fine_view = self.fl.evaluable_view(self.rst)
        self.rst_coarse_view = self.cl.mid

    def prolongate(self):
        """Prolongates from one Level to another

            Again the MultiLevelProvider hast to assure that the fine
            and coarse level has to be designed right.
        """
        self.ipl.eval(self.ipl_fine_views, self.ipl_coarse_views)

    def restringate(self):
        """Restringates from one Level to another

        """
        self.rst.eval(self.rst_fine_view, self.rst_coarse_view)

class MultiGridLevelProvider1D(object):
    """Contains the needed LevelTransition Provider

    Different cycles are provided, like the full multigrid cycle through the
    use of a char list. Note that the Smoother can also be a simple solver if
    the grid is coarse enough.
    """

    def __init__(self, levels, ipl_dict=None, rst_dict=None,
                 smth_dict=None):



        for k, v in levels.iteritems():
            assert_is_instance(v, MultiGridLevel1D,
                               "Not an MultiGridLevel1D object")
        for k, v in ipl_dict.iteritems():
            assert_is_instance(k, str, "Keys should be strings.")
            assert_is_instance(v, InterpolationStencil1D,
                               k+" is not an interpolation stencil.")
        for k, v in rst_dict.iteritems():
            assert_is_instance(k, str, "Keys should be strings")
            assert_is_instance(v, RestrictionStencil,
                               k+" is not a restriction stencil!")
        for k, v in smth_dict.iteritems():
            assert_is_instance(k, str, "Keys should be strings")
            assert_is_instance(v, Smoother,
                               k+" is not a smoother!")

        self.smth_dict = smth_dict
        self.levels = levels
        self._num_levels = len(levels)
        self.ipl_dict = ipl_dict
        self.rst_dict = rst_dict


    def do(self, **kwargs):
        """Takes order in string format and parse them into operations

        It follows the following notation, where k is the key of the level
        or smoother. the following keys are necessary for the different tasks

        smoothing : smoother=name of smoother or smoother itself or
                             the SmootherClass which is beeing used
                    smooth_n_times = ...
        interpolating : interpolation=name of interpolationstencil
                        from_level = name of the finer level
                        to_level = name of the coarser level
        restriction :   restriction=name of interpolationstencil
                        from_level = name of the finer level
                        to_level = name of the coarser level
        padding:        level = level or name of the level

        Note that each smoother object is bounded to a certain level,
        hence it is possible to use different smoothers on the same level,
        they have just to be in the smth_dict.
        """

        if kwargs["level"]:
            # padding
            self.levels[kwargs["level"]].pad()
        elif kwargs["smoother"]:
            # smooth
            self.smth_dict[kwargs["smoother"]].relax(kwargs["smooth_n_times"])
        elif kwargs["interpolation"]:
            arr_in = self.levels[kwargs["from_level"]]
            arr_out = self.levels[kwargs["to_level"]]
            self.ipl_dict[kwargs["interpolation"]].eval(arr_in, arr_out)
        elif kwargs["restriction"]:
            arr_in = self.levels[kwargs["from_level"]]
            arr_out = self.levels[kwargs["to_level"]]
            self.ipl_dict[kwargs["restriction"]].eval(arr_in, arr_out)
        else:
            raise NotImplementedError("I got my finger stuck in the bottle, "
                                      "the instructions were unclear")

# class MultiGridLevelProvider(object):
#     """
#     Summary
#     _______
#     Contains all interpolation and restriction operators.
#     """
#     def __init__(self, number_of_levels, interpolation_stencil_set, restriction_stencil_set):
#         self._nol = number_of_levels
#         assert_is_instance(interpolation_stencil_set, list)
#         assert_is_instance(restriction_stencil_set, list)
#         nipl = len(interpolation_stencil_set)
#         nrst = len(restriction_stencil_set)
#         self._nipl = nipl
#         self._nrst = nrst
#         assert_condition((self._nol == nipl or nipl == 1)
#                          and (self._nol == nrst or nrst == 1),
#                          "The number of interpolation and restriction operators"
#                          + " should be 1 or the number of levels")
#         self._ipls = interpolation_stencil_set
#         self._rsts = restriction_stencil_set
#
#         self._ipl_mult = []
#         self._rst_divi = []
#
#         for obj in restriction_stencil_set:
#             assert_is_instance(obj, np.ndarray, "One of the restriction"
#                                                 + "stencils is not an np.array")
#             self._rst_divi.append(obj.shape[0])
#         for obj in interpolation_stencil_set:
#             assert_is_instance(obj, np.ndarray, "One of the interpolation"
#                                                 + "stencils is not an np.array")
#         # es wird vermutlich eine eigene definition geben.
#         # fuer die interpolation finden wir nun heraus wie viele punkte hinzu kommen
#             self._ipl_mult.append(obj.shape[0])
#
#         # es sollte ueberlegt werden welches level gerade behandelt wird
#         # hiermit entscheide ich vorerst, dass 0 das feinste level ist
#
#
#         self._akt_level = 0
#
#     def set_akt_level(self, level):
#         """
#         returns the current level
#         Parameters
#         ----------
#         level : integer
#             indicates on which level the interpolation is done
#         """
#         assert_condition(level % 1 == 0 and level < self._nol,
#                          "level is too big or a float")
#         self._akt_level = level
#
#     def akt_ipl(self, level=None):
#         """
#         returns the current interpolation stencil
#         Parameters
#         ----------
#         level : integer
#             indicates on which level the interpolation is done
#         """
#         if level is not None:
#             self.set_akt_level(level)
#         if self._nipl == 1:
#             return self._ipls[0]
#         else:
#             return self._ipls[self._akt_level]
#
#     def interpolate(self, u_pad, level=None):
#         """
#         The main interpolation function,
#         if the interpolation stencil is bigger than the system stencil
#         the boundaries are computed wrong. Hence it is important to have
#         the padding of u right.
#         Parameters
#         ----------
#         u_pad : ndarray
#             the padded value region
#         level : integer
#             indicates on which level the interpolation is done
#         """
#         if level is not None:
#             self.set_akt_level(level)
#         # anhand des multiplikationsfaktors wird die groesse der zureckgegebenen
#         # matrix berrechnet
#         dim = u_pad.ndim
#         ipA = np.zeros(((np.asarray(u_pad.shape)-1)
#                         * np.asarray(self._ipl_mult))+1)
#         ipl = self.akt_ipl()
#
#         if dim == 1:
#             for i in range(self._ipl_mult[0]):
#                 ipA[i::self._ipl_mult[0]] = sig.convolve(u_pad, ipl[i], 'valid')
#         elif dim == 2:
#             for i in range(self._ipl_mult[0]):
#                 for j in range(self._ipl_mult[1]):
#                     ipA[i::self._ipl_mult[0], j::self._ipl_mult[1]] = \
#                         sig.convolve(u_pad, ipl[i, j], 'valid')
#         else:
#             print("Wer will den sowas!")
#
#
#
#     def restrict(self, u_pad, level=None):
#         """
#         The main restriction operator.
#         Parameters
#         ----------
#         u_pad : ndarray
#             the padded value region
#         level : integer
#             indicates on which level the interpolation is done
#         """
#         if level is not None:
#             self.set_akt_level(level)
#         #
#         # dim = u_pad.ndim
#         # if dim == 1:
#         #     repA = u_pad[]
