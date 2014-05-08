# coding=utf-8

import numpy as np
from pypint.multi_level_providers.multi_level_provider import MultiLevelProvider
from pypint.utilities import assert_is_instance, assert_condition
from pypint.plugins.multigrid.stencil import Stencil
from pypint.plugins.multigrid.level import IMultigridLevel
import scipy.signal as sig
import scipy.sparse as sprs
from pypint.plugins.multigrid.i_multigrid_smoother import IMultigridSmoother


class DirectSolverSmoother(IMultigridSmoother):
    """Takes the stencil and wraps the solver of stencil class, so that ist may
    may be used in the MultiGridProvider and put it into the level

    """
    def __init__(self, stencil, level, mod_rhs=False):
        """__init__ method"""
        assert_is_instance(stencil, Stencil, "A Stencil object is needed")
        assert_is_instance(level, IMultigridLevel, "Level should be "
                                                  "level instance")
        self.level = level
        self.solver = stencil.generate_direct_solver(level.mid.shape)
        self.stencil = stencil
        if mod_rhs:
            stencil.modify_rhs(level.evaluable_view(stencil), level.rhs)
        # print("**ESEL**", stencil.sp_matrix.todense())

    def relax(self):
        """ Just solves it, and puts the solution into self.level.mid
        """
        self.level.mid.reshape(-1)[:] = self.solver(self.level.rhs * (self.level.h**self.stencil.order))
        # print("kakao:", self.level.mid)


class SplitSmoother(IMultigridSmoother):
    """ A general Smoothing class which arises from splitting the main stencil

    This class of smoothers is easy to derive and really broad,
    it is statically linked to a certain level.
    """

    def __init__(self, l_plus, l_minus, level, *args, **kwargs):
        """init method of the split smoother

        l_plus and l_minus have to be centralized
        """


        assert_is_instance(l_plus, np.ndarray, "L plus has to be a np array")
        assert_is_instance(l_minus, np.ndarray, "L minus has to be a np array")
        assert_condition(l_plus.shape == l_minus.shape, TypeError,
                         "It is not an splitting")
        # check the level !has to be improved! inheritance must exist for
        # level class
        assert_is_instance(level, IMultigridLevel, "Not the right level")

        self.order = kwargs.get("order", 0)

        self.lvl = level
        self.l_plus = l_plus
        self.l_minus = l_minus
        self.st_minus = Stencil(l_minus)
        self.st_plus = Stencil(l_plus)

        self.l_plus_solver = self.st_plus.generate_direct_solver(self.lvl.mid.shape)
        # self.lvl_view_outer = level.evaluable_view(self.st_minus.b*2)
        #
        # grid = self.lvl_view_inner.shape
        # self.st_plus = Stencil(l_plus, grid=grid, solver="factorize")
        if level.modified_rhs is False:
            self.convolve_control = "valid"
            self.evaluable_view = level.evaluable_view(self.st_minus)
        else:
            print("huhu")
            self.convolve_control = "same"
            self.evaluable_view = level.mid

        super().__init__(l_plus.ndim, *args, **kwargs)

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
            self.lvl.mid.reshape(-1)[:] = self.l_plus_solver(self.lvl.rhs -
                                                             self.st_minus.eval_convolve(self.evaluable_view,
                                                                                         self.convolve_control))
        # the st_minus stencil contains in opposite to the usual stencil the factor self.h**2
            # self.lvl.mid.reshape(-1)[:] = \
            #     self.st_plus.solver(self.lvl.rhs
            #         - self.st_minus.eval_convolve(
            #             self.lvl_view_outer).reshape(-1)).reshape(
            #                 self.lvl_view_inner.shape)



class WeightedJacobiSmoother(IMultigridSmoother):
    """Implement a simple JaocbiSmoother , to test the SplitSmoother

    """

    def __init__(self, A_stencil, level, omega=0.5, computational_strategy_flag="matrix",**kwargs):
        """init

        """
        self.level = level
        self.omega = omega
        self.center_value = A_stencil.arr[tuple(A_stencil.center)]
        self.lvl_view = level.evaluable_view(A_stencil)

        if computational_strategy_flag == "matrix":
            # this branch needs the matrix R_w = (1 - w)I +wR_j
            # with R_j = D^-1 * (L+U)
            A = A_stencil.to_sparse_matrix(level.mid.shape)
            L = sprs.tril(-A, -1)
            U = sprs.triu(-A, 1)
            I = sprs.eye(level.mid.size, level.mid.size, 0, np.float64, "lil")
            self.D = self.center_value
            print("Matrices of the weighted Jacobian class: ")
            print("A :\n", A.todense())
            print("L :\n", L.todense())
            print("U :\n", U.todense())
            print("I :\n", I.todense())
            print("D :\n", self.D)

            self.R_w = (1.0-omega) * I + omega * (L + U) / self.D
            self.R_w = self.R_w.tocsc()
            self.relax = self._relax_matrix

        elif computational_strategy_flag == "convolve":
            # define a new stencil
            self.tmp = A_stencil.arr.copy()
            self.tmp[tuple(A_stencil.center)] *= (1.0 - 1.0/self.omega)
            self.stencil = Stencil(self.tmp, A_stencil.center)
            self.relax = self._relax_convolve

        elif computational_strategy_flag == "loop":
            self.stencil = A_stencil
            self.is_on_border = level.border_function_generator(A_stencil)
            self.relax = self._relax_loop
            # construct the stencil positions
        else:
            raise ValueError("Not a vaild flag, don't know how to compute")

        super().__init__(level.dim, **kwargs)



    def _relax_loop(self, n=1):
        """Does the jacobi relaxation step n times
            this function is meant to be compared with the other two
            implementations in order to check the implementation

            this implementation is really really slow, should just be used for
            tests
        """
        tmp = self.lvl_view.copy()
        flat_iter = self.lvl_view.flat
        # print("Stencil :", self.stencil.arr)
        # print("LevelView:", self.lvl_view)
        # print("relative Posistions :", self.stencil.relative_positions)
        # print("Size of lvl_view: ", self.lvl_view.size)
        for i in range(self.lvl_view.size):
            if not self.is_on_border(flat_iter.coords):
                # apply L_minus on u
                # print("The coords ", flat_iter.coords, "are not on the border")
                # print("And the relative coordinates used are:")
                my_sum = self.center_value * (1.0-1.0/self.omega) * self.lvl_view[flat_iter.coords]
                for st_pos in self.stencil.relative_positions_woc:
                    coords = tuple(np.asarray(flat_iter.coords) + np.asarray(st_pos))
                    # print("relative_coords: ", coords)
                    # print("coords_in_stencil: ", st_pos + self.stencil.center)
                    my_sum += self.stencil.arr[tuple(st_pos + self.stencil.center)] * \
                                self.lvl_view[coords]

                tmp[flat_iter.coords] = -my_sum * self.omega / self.center_value
                # print(self.lvl_view[flat_iter.coords])
            next(flat_iter)
        self.lvl_view.reshape(-1)[:] = tmp.reshape(-1)[:]

    def _relax_matrix(self, n=1):
        """ Using sparse matrix for the iteration steps

        """

        for i in range(n):
            self.level.mid.reshape(-1)[:] = self.R_w.dot(self.level.mid.reshape(-1)) \
                                            + self.omega * self.level.rhs / self.D

    def _relax_convolve(self, n=1):
        """Why bother, using simple convolution by defining a new stencil
           and use its convolution method.

        """

        self.level.mid.reshape(-1)[:] = \
            - self.stencil.eval_convolve(self.lvl_view) \
                * self.omega / self.center_value

