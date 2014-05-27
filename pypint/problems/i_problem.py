# coding=utf-8
"""
.. moduleauthor: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
import warnings
from collections import OrderedDict

import numpy as np

from pypint.plugins.implicit_solvers.find_root import find_root
from pypint.utilities import assert_is_callable, assert_is_instance, assert_is_in, class_name, assert_condition
from pypint.utilities.logging import LOG


class IProblem(object):
    """Basic interface for all problems of type :math:`u'(t,\\phi(t))=F(t,\\phi(t))`
    """

    valid_numeric_types = ['i', 'u', 'f', 'c']

    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        rhs_function_wrt_time : :py:class:`callable`
            Function describing the right hand side of the problem equation.
            Two arguments are required, the first being the time point :math:`t` and the second the time-dependent
            value :math:`\\phi(t)`.
        time_start : :py:class:`float`
            Start of the time interval to integrate over.
        time_end : :py:class:`float`
            End of the time interval to integrate over.
        dim : :py:class:`tuple`
            Number of spacial dimensions, i.e. number of degrees of freedom including shape of spacial points.
            The first elements denote the number of spacial dimensions (default=``1``) and the last the number of
            degrees of freedom (variables) at each spacial point.
            Defaults to ``(1, 1)``.
        strings : :py:class:`dict`
            *(optional)*
            String representation of problem for logging output.

            rhs_wrt_time : :py:class:`str`
                string representation of the right hand side w.r.t. time

        Examples
        --------
        >>> # default Problem
        >>> prob = IProblem()
        >>> prob.dim
        (1, 1)
        >>> # Problem with two spacial dimensions and one variable at each point
        >>> prob = IProblem(dim=(2, 3, 1))
        >>> prob.dim
        (2, 3, 1)
        >>> prob.spacial_dim
        (2, 3)
        >>> prob.num_spacial_points
        6
        >>> prob.dofs_per_point
        1
        """
        self._rhs_function_wrt_time = None
        if 'rhs_function_wrt_time' in kwargs:
            self.rhs_function_wrt_time = kwargs['rhs_function_wrt_time']

        self._numeric_type = np.float
        if 'numeric_type' in kwargs:
            self.numeric_type = kwargs['numeric_type']

        self._dim = (1, 1)
        if kwargs.get('dim') is not None:
            assert_is_instance(kwargs['dim'], tuple, descriptor="Spacial Degrees of Freedom", checking_obj=self)
            for index in range(0, len(kwargs['dim']) - 1):
                assert_is_instance(kwargs['dim'][index], int, descriptor="Number of Spacial Points at axis %d" % index,
                                   checking_obj=self)
            assert_is_instance(kwargs['dim'][-1], int, descriptor="Variables at each Spacial Point", checking_obj=self)
            self._dim = kwargs['dim']

        self._strings = {
            'rhs_wrt_time': None
        }
        if 'strings' in kwargs:
            if 'rhs_wrt_time' in kwargs['strings']:
                self._strings['rhs_wrt_time'] = kwargs['strings']['rhs_wrt_time']

        self._count_rhs_eval = 0

    def evaluate_wrt_time(self, time, phi_of_time, **kwargs):
        """Evaluates given right hand side at given time and with given time-dependent value.

        Parameters
        ----------
        time : :py:class:`float`
            Time point :math:`t`
        phi_of_time : :py:class:`numpy.ndarray`
            Time-dependent data.
        partial : :py:class:`str` or :py:class:`None`
            *(optional)*
            Specifying whether only a certain part of the problem function should be evaluated.
            E.g. useful for semi-implicit SDC where the imaginary part of the function is explicitly evaluated and
            the real part of the function implicitly.
            Usually it is one of :py:class:`None`, ``impl`` or ``expl``.

        Returns
        -------
        rhs_value : :py:class:`numpy.ndarray`

        Raises
        ------
        ValueError :
            if ``time`` or ``phi_of_time`` are not of correct type.
        """
        assert_is_instance(time, float, descriptor="Time Point", checking_obj=self)
        assert_is_instance(phi_of_time, np.ndarray, descriptor="Data Vector", checking_obj=self)
        if kwargs.get('partial') is not None:
            assert_is_instance(kwargs['partial'], str, descriptor="Partial Descriptor", checking_obj=self)
        self._count_rhs_eval += 1
        return np.zeros(self.dim, dtype=self.numeric_type)

    def implicit_solve(self, next_x, func, method="hybr", **kwargs):
        """A solver for implicit equations.

        Finds the implicitly defined :math:`x_{i+1}` for the given right hand side function :math:`f(x_{i+1})`, such
        that :math:`x_{i+1}=f(x_{i+1})`.


        Parameters
        ----------
        next_x : :py:class:`numpy.ndarray`
            A starting guess for the implicitly defined value.
        rhs_call : :py:class:`callable`
            The right hand side function depending on the implicitly defined new value.
        method : :py:class:`str`
            *(optional, default=``hybr``)*
            Method fo the root finding algorithm. See `scipy.optimize.root
            <http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html#scipy.optimize.root>` for
            details.

        Returns
        -------
        next_x : :py:class:`numpy.ndarray`
            The calculated new value.

        Raises
        ------
        ValueError :

            * if ``next_x`` is not a :py:class:`numpy.ndarray` of shape :py:attr:`.IProblem.dim`
            * if ``fun`` is not :py:class:`callable`
            * if computed solution is not a `:py:class:`numpy.ndarray`

        UserWarning :
            If the implicit solver did not converged, i.e. the solution object's ``success`` is not :py:class:`True`.
        """
        assert_is_instance(next_x, np.ndarray, descriptor="Initial Guess", checking_obj=self)
        assert_is_callable(func, descriptor="Function of RHS for Implicit Solver", checking_obj=self)
        sol = find_root(fun=func, x0=next_x.reshape(-1), method=method)
        if not sol.success:
            warnings.warn("Implicit solver did not converged.")
            LOG.debug("sol.x: %s" % sol.x)
            LOG.error("Implicit solver failed: %s" % sol.message)
        else:
            assert_is_instance(sol.x, np.ndarray, descriptor="Solution", checking_obj=self)
        return sol.x.reshape(self.dim_for_time_solver)

    @property
    def rhs_function_wrt_time(self):
        """Accessor for the right hand side function.

        Parameters
        ----------
        function : :py:class:`callable`
            Function of the right hand side of :math:`u'(t,x)=F(t,\\phi_t)`

        Returns
        -------
        rhs_function : :py:class:`callable`
            Function of the right hand side.
        """
        return self._rhs_function_wrt_time

    @rhs_function_wrt_time.setter
    def rhs_function_wrt_time(self, function):
        assert_is_callable(function, descriptor="Function of the RHS w.r.t Time", checking_obj=self)
        self._rhs_function_wrt_time = function

    @property
    def rhs_evaluations(self):
        return self._count_rhs_eval

    @rhs_evaluations.deleter
    def rhs_evaluations(self):
        self._count_rhs_eval = 0

    @property
    def numeric_type(self):
        """Accessor for the numerical type of the problem values.

        Parameters
        ----------
        numeric_type : :py:class:`numpy.dtype`
            Usually it is :py:class:`numpy.float64` or :py:class:`numpy.complex16`

        Returns
        -------
        numeric_type : :py:class:`numpy.dtype`

        Raises
        ------
        ValueError :
            If ``numeric_type`` is not a :py:class:`numpy.dtype`.
        """
        return self._numeric_type

    @numeric_type.setter
    def numeric_type(self, numeric_type):
        numeric_type = np.dtype(numeric_type)
        assert_is_in(numeric_type.kind, IProblem.valid_numeric_types, elem_desc="Numeric Type", list_desc="Valid Types",
                     checking_obj=self)
        self._numeric_type = numeric_type

    @property
    def dim(self):
        """Read-only accessor for the spacial degrees of freedom of the problem

        Returns
        -------
        dofs : :py:class:`tuple`
            First elements denotes shape of spacial points; the last element the number degrees of freedom (i.e.
            variables) at each spacial point.
        """
        return self._dim

    @property
    def dim_for_time_solver(self):
        """Dimension of array for Time Solvers

        This shape is used for the arrays of time solvers, which do not need to know the spacial shape of the spacial
        points.

        Returns
        -------
        dim_for_time_solver : :py:class:`tuple`
            First element is the total number of spacial points (:py:attr:`.num_spacial_points`) and the second element
            the number of variables per spacial point (:py:attr:`.dofs_per_point`).
        """
        return self.num_spacial_points, self.dofs_per_point

    @property
    def spacial_dim(self):
        """Shape of spacial points

        Returns
        -------
        spacial_dim : :py:class:`tuple`
        """
        return self.dim[0:-1]
    
    @property
    def num_spacial_points(self):
        """Total number of spacial points

        Returns
        -------
        num_spacial_points : :py:class:`int`
            product of the elements of :py:attr:`.spacial_dim`
        """
        return np.asarray(self.spacial_dim, dtype=np.int).prod()

    @property
    def dofs_per_point(self):
        """Variables / Degrees of Freedom at each spacial point

        dofs_per_point : :py:class:`int`
        """
        return self.dim[-1]

    def print_lines_for_log(self):
        _lines = OrderedDict()
        if self._strings['rhs_wrt_time'] is not None:
            _lines['Formula w.r.t. Time'] = r"u(t, \phi(t)) = %s" % self._strings['rhs_wrt_time']
        _lines['DOFs'] = "{:s}".format(self.dim)
        return _lines

    def __str__(self):
        if self._strings['rhs_wrt_time'] is not None:
            _outstr = r"u'(t,\phi(t))=%s" % self._strings['rhs_wrt_time']
        else:
            _outstr = r"%s" % class_name(self)
        _outstr += r", DOFs={:s}".format(self.dim)
        return _outstr


__all__ = ['IProblem']
