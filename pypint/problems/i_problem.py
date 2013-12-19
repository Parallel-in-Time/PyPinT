# coding=utf-8
"""

.. moduleauthor: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

import numpy as np
from pypint.plugins.implicit_solvers.find_root import find_root
from pypint import LOG
from pypint.utilities.tracing import assert_is_callable, assert_is_instance, critical_assert


class IProblem(object):
    """
    Summary
    -------
    Basic interface for all problems of type :math:`u'(t,\\phi(t))=F(t,\\phi(t))`

    Parameters
    ----------
    kwargs : dict

        ``function`` : function pointer | lambda
            Function describing the right hand side of the problem equation.
            Two arguments are required, the first being the time point :math:`t` and the second
            the time-dependent value :math:`\\phi(t)`.

        ``time_start`` : float
            Start of the time interval to integrate over.

        ``time_end`` : float
            End of the time interval to integrate over.

        ``exact_function`` : function pointer | lambda
            (optional)
            If specified, this function describes the exact solution of the given problem.

        ``strings`` : dict
            (optional)

            ``rhs`` : string
                (optional)
                String representation of the right hand side function for logging output.

            ``exact`` : string
                (optional)
                String representation of the exact solution for logging output.
    """
    def __init__(self, *args, **kwargs):
        self._function = kwargs["function"] if "function" in kwargs else None
        self._time_start = kwargs["time_start"] if "time_start" in kwargs else None
        self._time_end = kwargs["time_end"] if "time_end" in kwargs else None
        self._exact_function = kwargs["exact_function"] if "exact_function" in kwargs else None
        self._numeric_type = np.float
        self._strings = {
            "rhs": None,
            "exact": None
        }
        if "strings" in kwargs:
            if "rhs" in kwargs["strings"]:
                self._strings["rhs"] = kwargs["strings"]["rhs"]
            if "exact" in kwargs["strings"]:
                self._strings["exact"] = kwargs["strings"]["exact"]

    def evaluate(self, time, phi_of_time, partial=None):
        """
        Summary
        -------
        Evaluates given right hand side at given time and with given time-dependent value.

        Parameters
        ----------
        time : float
            Time point :math:`t`

        phi_of_time : ``numpy.ndarray``
            Time-dependent data.

        partial : str
            Specifying whether only a certain part of the problem function should be evaluated.
            E.g. useful for semi-implicit SDC where the imaginary part of the function is explicitly evaluated and
            the real part of the function implicitly.
            Usually ``partial`` is one of ``None``, ``impl`` or ``expl``.

        Returns
        -------
        RHS value : numpy.ndarray

        Raises
        ------
        ValueError
            if ``time`` or ``phi_of_time`` are not of correct type.
        """
        assert_is_instance(time, float, "Time must be given as a floating point number.", self)

    def implicit_solve(self, next_x, func, method="hybr"):
        """
        Summary
        -------
        A solver for implicit equations.

        Extended Summary
        ----------------
        Finds the implicitly defined :math:`x_{i+1}` for the given right hand side function :math:`f(x_{i+1})`, such
        that :math:`x_{i+1}=f(x_{i+1})`.

        Parameters
        ----------
        next_x : numpy.ndarray
            A starting guess for the implicitly defined value.

        rhs_call : callable
            The right hand side function depending on the implicitly defined new value.

        method : str
            Method fo the root finding algorithm. See `scipy.optimize.root`_ for details.

        Returns
        -------
        next_x : numpy.ndarray
            The calculated new value.

        .. _scipy.optimize.root: http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html#scipy.optimize.root
        """
        assert_is_instance(next_x, np.ndarray, "Need a numpy.ndarray.", self)
        assert_is_callable(func, "Need a callable function.", self)
        sol = find_root(fun=func, x0=next_x, method=method)
        # LOG.debug("Root is: {:s}, {:s}".format(sol.x, sol.x.dtype))
        if not sol.success:
            LOG.debug("sol.x: " + str(sol.x))
            LOG.error("Implicit solver failed: {:s}".format(sol.message))
        return sol.x

    def exact(self, time, phi_of_time):
        """
        Summary
        -------
        Evaluates given exact solution function at given time and with given time-dependent data.

        Parameters
        ----------
        time : float
            Time point :math:`t`

        phi_of_time : ``numpy.ndarray``
            Time-dependent data.

        Returns
        -------
        exact solution : numpy.ndarray
        """
        return self.exact_function(time, phi_of_time)

    def has_exact(self):
        """
        Summary
        -------
        Convenience accessor for exact solution.

        Returns
        -------
         : boolean
            ``True`` if exact solution was given, ``False`` otherwise
        """
        return self.exact_function is not None

    @property
    def function(self):
        """
        Summary
        -------
        Accessor for the right hand side function.

        Parameters
        ----------
        function : function pointer | lambda
            Function of the right hand side of :math:`u'(t,x)=F(t,\\phi_t)`

        Returns
        -------
        rhs function : function_pointer | lambda
            Function of the right hand side.
        """
        return self._function

    @function.setter
    def function(self, function):
        self._function = function

    @property
    def time_start(self):
        """
        Summary
        -------
        Accessor for the time interval's start.

        Parameters
        ----------
        interval start : float
            Start point of the time interval.

        Returns
        -------
        interval start : float
            Start point of the time interval.
        """
        return self._time_start

    @time_start.setter
    def time_start(self, time_start):
        self._time_start = time_start

    @property
    def time_end(self):
        """
        Summary
        -------
        Accessor for the time interval's end.

        Parameters
        ----------
        interval end : float
            End point of the time interval.

        Returns
        -------
        interval end : float
            End point of the time interval.
        """
        return self._time_end

    @time_end.setter
    def time_end(self, time_end):
        self._time_end = time_end

    @property
    def exact_function(self):
        """
        Summary
        -------
        Accessor for exact solution.

        Parameters
        ----------
        function : function pointer | lambda
            Function of the exact solution of :math:`u'(t,x)=F(t,\\phi_t)`

        Returns
        -------
        rhs function : function_pointer | lambda
            Function of the exact solution.
        """
        return self._exact_function

    @exact_function.setter
    def exact_function(self, exact_function):
        self._exact_function = exact_function

    @property
    def numeric_type(self):
        return self._numeric_type

    def __str__(self):
        str = ""
        if self._strings["rhs"] is not None:
            str = r"u'(t,\phi(t))={:s}".format(self._strings["rhs"])
        else:
            str = r"{:s}".format(self.__class__.__name__)
        str += r", t \in [{:.2f}, {:.2f}]".format(self.time_start, self.time_end)
        return str
