# coding=utf-8
"""

.. moduleauthor: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
import warnings
from collections import OrderedDict

import numpy as np

from pypint.plugins.implicit_solvers.find_root import find_root
from pypint.utilities import assert_is_callable, assert_is_instance, assert_is_in, class_name
from pypint.utilities.logging import LOG


class IProblem(object):
    """Basic interface for all problems of type :math:`u'(t,\\phi(t))=F(t,\\phi(t))`
    """

    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        function : :py:class:`callable`
            Function describing the right hand side of the problem equation.
            Two arguments are required, the first being the time point :math:`t` and the second the time-dependent
            value :math:`\\phi(t)`.

        time_start : :py:class:`float`
            Start of the time interval to integrate over.

        time_end : :py:class:`float`
            End of the time interval to integrate over.

        dim : :py:class:`int`
            Number of spacial dimensions.

        rhs: :py:class:`str`
            *(optional)*
            String representation of the right hand side function for logging output.
        """
        self._function = None
        if "function" in kwargs:
            self.function = kwargs["function"]

        self._time_start = 0.0
        if "time_start" in kwargs:
            self.time_start = kwargs["time_start"]

        self._time_end = 1.0
        if "time_end" in kwargs:
            self.time_end = kwargs["time_end"]

        self._numeric_type = np.float
        if "numeric_type" in kwargs:
            self.numeric_type = kwargs["numeric_type"]

        self._dim = kwargs["dim"] if "dim" in kwargs else 1

        self._strings = {
            "rhs": None,
            "exact": None
        }
        if "strings" in kwargs:
            if "rhs" in kwargs["strings"]:
                self._strings["rhs"] = kwargs["strings"]["rhs"]

    def evaluate(self, time, phi_of_time, partial=None):
        """Evaluates given right hand side at given time and with given time-dependent value.

        Parameters
        ----------
        time : :py:class:`float`
            Time point :math:`t`

        phi_of_time : :py:class:`numpy.ndarray`
            Time-dependent data.

        partial : :py:class:`str` or :py:class:`None`
            Specifying whether only a certain part of the problem function should be evaluated.
            E.g. useful for semi-implicit SDC where the imaginary part of the function is explicitly evaluated and
            the real part of the function implicitly.
            Usually it is one of :py:class:`None`, ``impl`` or ``expl``.

        Returns
        -------
        RHS value : :py:class:`numpy.ndarray`

        Raises
        ------
        ValueError :
            if ``time`` or ``phi_of_time`` are not of correct type.
        """
        assert_is_instance(time, float, descriptor="Time Point", checking_obj=self)
        assert_is_instance(phi_of_time, np.ndarray, descriptor="Data Vector", checking_obj=self)
        return np.zeros(self.dim, dtype=self.numeric_type)

    def implicit_solve(self, next_x, func, method="hybr"):
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

            * if ``next_x`` is not a :py:class:`numpy.ndarray`
            * if ``fun`` is not :py:class:`callable`
            * if computed solution is not a `:py:class:`numpy.ndarray`

        UserWarning :
            If the implicit solver did not converged, i.e. the solution object's ``success`` is not :py:class:`True`.
        """
        assert_is_instance(next_x, np.ndarray, descriptor="Initial Guess", checking_obj=self)
        assert_is_callable(func, descriptor="Function of RHS", checking_obj=self)
        sol = find_root(fun=func, x0=next_x, method=method)
        if not sol.success:
            warnings.warn("Implicit solver did not converged.")
            LOG.debug("sol.x: " + str(sol.x))
            LOG.error("Implicit solver failed: {:s}".format(sol.message))
        else:
            assert_is_instance(sol.x, np.ndarray, descriptor="Solution", checking_obj=self)
        return sol.x

    @property
    def function(self):
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
        return self._function

    @function.setter
    def function(self, function):
        assert_is_callable(function, checking_obj=self)
        self._function = function

    @property
    def time_start(self):
        """Accessor for the time interval's start.

        Parameters
        ----------
        interval_start : :py:class:`float`
            Start point of the time interval.

        Returns
        -------
        interval_start : :py:class:`float`
            Start point of the time interval.
        """
        return self._time_start

    @time_start.setter
    def time_start(self, time_start):
        self._time_start = time_start

    @property
    def time_end(self):
        """Accessor for the time interval's end.

        Parameters
        ----------
        interval_end : :py:class:`float`
            End point of the time interval.

        Returns
        -------
        interval_end : :py:class:`float`
            End point of the time interval.
        """
        return self._time_end

    @time_end.setter
    def time_end(self, time_end):
        self._time_end = time_end

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
        _valid_types = ['i', 'u', 'f', 'c']
        assert_is_in(numeric_type.kind, _valid_types, elem_desc="Numeric Type", list_desc="Valid Types",
                     checking_obj=self)
        self._numeric_type = numeric_type

    @property
    def dim(self):
        """Read-only accessor for the spacial dimension of the problem

        Returns
        -------
        spacial_dimension : :py:class:`int`
        """
        return self._dim

    def print_lines_for_log(self):
        _lines = OrderedDict()
        if self._strings['rhs'] is not None:
            _lines['Formula'] = 'u(t, \phi(t)) = %s' % self._strings["rhs"]
        _lines['Interval'] = '[{:.3f}, {:.3f}]'.format(self.time_start, self.time_end)
        return _lines

    def __str__(self):
        if self._strings["rhs"] is not None:
            _outstr = r"u'(t,\phi(t))={:s}".format(self._strings["rhs"])
        else:
            _outstr = r"{:s}".format(class_name(self))
        _outstr += r", t \in [{:.2f}, {:.2f}]".format(self.time_start, self.time_end)
        return _outstr


__all__ = ['IProblem']
