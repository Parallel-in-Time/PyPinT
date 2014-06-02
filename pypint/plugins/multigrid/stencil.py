# coding=utf-8
import numpy as np
import scipy.signal as sig
import scipy.sparse as sprs
import scipy.sparse.linalg as spla
import functools as ft
from pypint.utilities import assert_is_callable, assert_is_instance, assert_condition
from pypint.plugins.multigrid.i_multigrid_level import IMultigridLevel
from pypint.utilities.logging import LOG
from pypint.utilities import func_name


class Stencil(object):
    """
    Summary
    -------
    a class which knows its center

    Has a lot to over, like the conversion to a sparse matrix,
    a wrapper around the solvers of the sparse linalg scipy class,
    and with self.b it has the appropriate information for the
    handling of the ghostcells.
    """
    def __init__(self, arr, center=None, order=1, **kwargs):
        assert_is_instance(arr, np.ndarray, "the array is not a numpy array")
        if center is None:
            self.center = np.array(np.floor(np.asarray(arr.shape)*0.5),
                                   dtype=np.int)
        else:
            assert_is_instance(center, np.ndarray,
                               "the center is not a np array")
            assert_condition(arr.ndim == center.size, ValueError,
                             "center does not match with stencil array")
            self.center = np.array(center, dtype=np.int)
        self.arr = arr
        self.dim = arr.ndim
        self.order = order
        # compute borders
        self.b = [[0.0, 0.0]]*self.dim
        for i in range(self.dim):
            left = arr.shape[i] - self.center[i] - 1
            right = arr.shape[i] - left - 1
            self.b[self.dim - i - 1] = [right, left]
        self.b = np.asarray(self.b)
        # check if a grid is given
        self._grid = None

        if 'grid' in kwargs:
            self._grid = tuple(kwargs['grid'])
        else:
            self._grid = tuple([3]*self.dim)
        # construct stencil positions and relative stencil positions
        self.positions = []
        self.relative_positions = []
        self.relative_positions_woc = []
        flat_iter = self.arr.flat

        # dear future me this hack does not need to be refactored
        # because next(flat_iterator) is called to early
        # and flat_iterator.coords does not work properly hence next is called
        # manually

        for i in range(self.arr.size):
            self.positions.append(flat_iter.coords)
            self.relative_positions.append(tuple(np.asarray(flat_iter.coords) - self.center))
            if not (self.center == np.asarray(flat_iter.coords)).all():
                self.relative_positions_woc.append(tuple(np.asarray(flat_iter.coords) - self.center))
            next(flat_iter)
        # relative positions without center
        # construct sparse matrix
        self.sp_matrix = self.to_sparse_matrix(self._grid)
        # check which solver should be used
        if kwargs.get('solver') is None or kwargs.get('solver') == "richardson":
            self.solver = self.richardson_solver
            self.solver_info = "Richardson Solver"
            self.solver_type = "iterative"
        elif kwargs['solver'] == 'factorize':
            self.sp_matrix = self.sp_matrix.tocsc()
            self.solver = self.generate_direct_solver()
            self.solver_info = "Direct Fast Solver through factorization"
            self.solver_type = "factorized"
        elif isinstance(kwargs['solver'], str):
            self.solver = ft.partial(self.iterative_solver_list,
                                     self, kwargs["solver"])
            self.solver_info = "Iterative solver of type" + kwargs['solver']
            self.solver_type = "iterative"
        elif callable(kwargs['solver']):
            # this is the case of user solver
            self.solver = kwargs['solver']
            if kwargs['solver_info'] is None:
                self.solver_info = " "
        else:
            raise TypeError("this solver is unknown!")

        # one needs reversed arr for  convolution operator
        self.reverse_slice = []
        for i in range(self.dim):
            self.reverse_slice.append(slice(None, None, -1))

        self.reversed_arr = self.arr[self.reverse_slice]

    @property
    def num_nodes(self):
        """Accessor for the number of desired integration nodes.

        Returns
        -------
        number of nodes : :py:class:`int`
            The number of desired and/or computed integration nodes.

        Notes
        -----
        Specializations of this interface might override this accessor.
        """
        return self._grid

    @property
    def grid(self):
        """grid getter

        """
        return self._grid

    @grid.setter
    def grid(self, grd):
        """Grid setter

        A new Grid implicates a whole new stencil matrix and if
        the solver constucted from the factorization one needs also a new
        factorization
        """
        self._grid = grd
        self.sp_matrix = self.to_sparse_matrix(grd)
        if self.solver_type == "factorized":
            self.solver = self.generate_direct_solver(grd)


    def richardson_solver(self, rhs, options):
        """Simple richardson Iteration

        """
        pass

    def generate_direct_solver(self, grid=None):
        """Generates direct solver from a LU factorization of the sparse matrix

        """
        if grid is None:
            # LOG.debug("Generate Solver for internal Spare Matrix: %s" % self.sp_matrix)
            solver = spla.factorized(self.sp_matrix)
        else:
            # LOG.debug("Generate Solver for given Grid %s" % (grid,))
            sp_matrix = self.to_sparse_matrix(grid, "csc")
            # LOG.debug("  with Sparse Matrix: %s" % sp_matrix.todense())
            # print("Jahier\n", sp_matrix.todense())
            # print("Jahier.shape\n", sp_matrix.todense().shape)
            solver = spla.factorized(sp_matrix)
        return solver

    def eval_convolve(self, array_in, convolve_control="valid"):
        """Evaluate via scipy.signal.convolve

        Parameters
        ----------
        array_in : ndarray
            array to convolve
        array_out : ndarray
            array to storage the result

        Examples
        --------
        It is possible to use this as just an evaluation funktion like Ax=b
        otherwise it is also possible to use it as an operator
        Ax=b:
        stencil.eval_convolve(level.mid, "same")
        as operator, where the boundary is taken into account:
        stencil.eval_convolve(level.evaluate_view(stencil),"valid" )
        """
        # LOG.debug("%s on %s with stencil %s" % (func_name(self), array_in, self.reversed_arr))
        _out = sig.convolve(array_in, self.reversed_arr, convolve_control)
        # LOG.debug("  ==> %s" % _out)
        return _out

    def eval_sparse(self, array_in, array_out, sp_matrix=None):
        """Evaluate via the sparse matrix

        Parameters
        ----------
        array_in : ndarray
            array to apply to
        array_out : ndarray
            array to storage the result
        """
        if sp_matrix is None:
            sp_matrix = self.to_sparse_matrix(array_in.shape, "csc")
            # print("usually:", sp_matrix.todense())
        array_out[:] = sp_matrix.dot(array_in.reshape(-1)).reshape(array_out.shape)


    def centered_stencil(self):
        """ use zero padding to put the center into the center

        """

        # compute the shape of the new stencil
        shp = self.arr.shape
        shp = tuple(max(i, j-(i+1))*2 + 1 for i, j in zip(self.center, shp))
        # print("New Shape :", shp)
        # generate the stencil in the right shape
        S = np.zeros(shp)
        # embed the stencil into the bigger stencil in order to place the center
        # into the center
        slc = []
        for c, shp_arr, shp_s in zip(self.center, self.arr.shape, shp):
            if c < shp_arr/2:
                slc.append(slice(shp_s - shp_arr, None))
            else:
                slc.append(slice(0, -(shp_s - shp_arr)))

        # print(slc)
        S[slc] = self.arr[:]
        # print("The Stencil")
        # print(self.arr)
        # print("Centered stencil")
        # print(S)
        return S

    def to_sparse_matrix(self, grid, format=None):
        """constructs a scipy dia sparse matrix

        This algorithm is, besides the embedding in the first few lines,
        taken from `PyAMG`_

        .. epigraph:

            Copyright (c) 2008, PyAMG Developers
            All rights reserved.

            Redistribution and use in source and binary forms, with or without
            modification, are permitted provided that the following conditions are
            met:

            * Redistributions of source code must retain the above copyright
              notice, this list of conditions and the following disclaimer.

            * Redistributions in binary form must reproduce the above
              copyright notice, this list of conditions and the following
              disclaimer in the documentation and/or other materials provided
              with the distribution.

            * Neither the name of the PyAMG Developers nor the names of any
              contributors may be used to endorse or promote products derived
              from this software without specific prior written permission.

            THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
            "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
            LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
            A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
            OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
            SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
            LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
            DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
            THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
            (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
            OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

        Parameters
        ----------
        grid : tuple
            tuple containing the N grid dimensions
        format : string
            sparse matrix format to return , e.g. "csr", "coo", etc.

        .. _PyAMG: https://github.com/pyamg/pyamg
        """
        S = self.centered_stencil()
        # print("grid :")

        grid = tuple(grid)
        # print(grid)
        if not (np.asarray(S.shape) % 2 == 1).all():
            raise ValueError('all stencil dimensions must be odd')

        assert_condition(len(grid) == np.rank(S), ValueError,
                         'stencil rank must equal number of grid dimensions')
        assert_condition(min(grid) >= 1, ValueError,
                         'grid dimensions must be positive')

        N_v = np.prod(grid)     # number of vertices in the mesh
        N_s = (S != 0).sum()    # number of nonzero stencil entries

        # diagonal offsets
        diags = np.zeros(N_s, dtype=int)

        # compute index offset of each dof within the stencil
        strides = np.cumprod([1] + list(reversed(grid)))[:-1]
        indices = tuple(i.copy() for i in S.nonzero())
        for i,s in zip(indices,S.shape):
            i -= s // 2
        for stride,coords in zip(strides, reversed(indices)):
            diags += stride * coords

        #
        data = S[S != 0].repeat(N_v).reshape(N_s, N_v)
        indices = np.vstack(indices).T

        # zero boundary connections
        for index,diag in zip(indices,data):
            diag = diag.reshape(grid)
            for n,i in enumerate(index):
                if i > 0:
                    s = [ slice(None) ]*len(grid)
                    s[n] = slice(0,i)
                    diag[s] = 0
                elif i < 0:
                    s = [ slice(None) ]*len(grid)
                    s[n] = slice(i,None)
                    diag[s] = 0

        # remove diagonals that lie outside matrix
        mask = abs(diags) < N_v
        if not mask.all():
            diags = diags[mask]
            data  = data[mask]

        # sum duplicate diagonals
        if len(np.unique(diags)) != len(diags):
            new_diags = np.unique(diags)
            new_data  = np.zeros( (len(new_diags),data.shape[1]), dtype=data.dtype)
            for dia,dat in zip(diags,data):
                n = np.searchsorted(new_diags,dia)
                new_data[n,:] += dat

            diags = new_diags
            data  = new_data

        return sprs.dia_matrix((data,diags), shape=(N_v, N_v)).asformat(format)

    def iterative_solver_list(self, which, rhs, *args):
        """Solves the linear problem Ab = x using the sparse matrix

            Parameters
            ----------
            rhs : ndarray
                the right hand side
            which : string
                choose which solver is used
                    bicg(A, b[, x0, tol, maxiter, xtype, M, ...])
                        Use BIConjugate Gradient iteration to solve A x = b

                    bicgstab(A, b[, x0, tol, maxiter, xtype, M, ...])
                        Use BIConjugate Gradient STABilized iteration to solve A x = b

                    cg(A, b[, x0, tol, maxiter, xtype, M, callback])
                        Use Conjugate Gradient iteration to solve A x = b

                    cgs(A, b[, x0, tol, maxiter, xtype, M, callback])
                        Use Conjugate Gradient Squared iteration to solve A x = b

                    gmres(A, b[, x0, tol, restart, maxiter, ...])
                        Use Generalized Minimal RESidual iteration to solve A x = b.

                    lgmres(A, b[, x0, tol, maxiter, M, ...])
                        Solve a matrix equation using the LGMRES algorithm.

                    minres(A, b[, x0, shift, tol, maxiter, ...])
                        Use MINimum RESidual iteration to solve Ax=b

                    qmr(A, b[, x0, tol, maxiter, xtype, M1, M2, ...])
                        Use Quasi-Minimal Residual iteration to solve A x = b
        """
        if which == 'bicg':
            return spla.bicg(self.sp_matrix, rhs, args)
        elif which == "cg":
            return spla.cg(self.sp_matrix, rhs, args)
        elif which == "bicgstab":
            return spla.bicgstab(self.sp_matrix, rhs, args)
        elif which == "cgs":
            return spla.cgs(self.sp_matrix, rhs, args)
        elif which == "gmres":
            return spla.gmres(self.sp_matrix, rhs, args)
        elif which == "lgmres":
            return spla.lgmres(self.sp_matrix, rhs, args)
        elif which == "qmr":
            return spla.qmr(self.sp_matrix, rhs, args)
        else:
            raise NotImplementedError("this solver is unknown")

    def modify_rhs(self, level):
        """ Modifies rhs

        Parameters
        ----------
        level: np.ndarray
        """
        if level.modified_rhs is False:
            u = level.evaluable_view(self)
            rhs = level.rhs
            # print("here from modify your right hand side:")
            # print("u", u)
            # print("rhs", rhs)

            if self.dim == 1:
                # LOG.debug("Modifying RHS")
                # left side
                for i in range(self.center[0]):
                    rhs[i] -= np.dot(self.arr[i:self.center[0]],
                                     u[0:self.center[0]-i])
                # the same for the right side
                til = self.arr.size - self.center[0] - 1

                for i in range(til):
                    rhs[-til] -= np.dot(self.arr[-til + i:], u[-til: u.size - i])

            elif self.dim == 2:
                temp_arr = np.copy(level.evaluable_view(self))
                temp_arr[level.mid_slice] = 0.0
                level.rhs[:] = level.rhs[:] - sig.convolve(temp_arr, self.reversed_arr, 'valid')

            elif self.dim == 3:
                raise NotImplementedError("Sure I will do it , like, really soon")
            else:
                raise NotImplementedError("No one needs more than 3 dimensions")

            level.modified_rhs = True

    def l_plus_jacobi(self, omega):
        l_plus = np.zeros(self.arr.shape)
        l_plus[self.center] = self.arr[self.center] / omega
        return l_plus

    def l_minus_jacobi(self, omega):
        l_minus = np.copy(self.arr)
        l_minus[self.center] *= (1.0 - 1.0 / omega)
        return l_minus
