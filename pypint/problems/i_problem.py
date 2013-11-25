# coding=utf-8
"""

.. moduleauthor: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""


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
        if "function" in kwargs:
            self._function = kwargs["function"]
        else:
            self._exact = None

        if "time_start" in kwargs:
            self._time_start = kwargs["time_start"]
        else:
            self._time_start = None

        if "time_end" in kwargs:
            self._time_end = kwargs["time_end"]
        else:
            self._time_end = None

        if "exact_function" in kwargs:
            self._exact_function = kwargs["exact_function"]
        else:
            self._exact_function = None

        self._strings = {
            "rhs": None,
            "exact": None
        }
        if "strings" in kwargs:
            if "rhs" in kwargs["strings"]:
                self._strings["rhs"] = kwargs["strings"]["rhs"]
            if "exact" in kwargs["strings"]:
                self._strings["exact"] = kwargs["strings"]["exact"]

    def evaluate(self, time, phi_of_time):
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

        Returns
        -------
        RHS value : numpy.ndarray
        """
        return self.function(time, phi_of_time)

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

    def __str__(self):
        str = ""
        if self._strings["rhs"] is not None:
            str = r"u'(t,\phi(t))={:s}".format(self._strings["rhs"])
        else:
            str = r"{:s}".format(self.__class__.__name__)
        str += r", t \in [{:.2f}, {:.2f}]".format(self.time_start, self.time_end)
        return str
