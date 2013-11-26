# coding=utf-8
"""

.. moduleauthor: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from pypint.problems.i_problem import IProblem


class IInitialValueProblem(IProblem):
    """
    Summary
    -------
    Basic interface for initial value problems.

    Parameters
    ----------
    In addition to the ones of :py:class:`.IProblem` the following options are supported:

    kwargs : dict
        ``initial_value`` : float | ``numpy.ndarray``
            Initial value of :math:`u(t_0,\\phi(t_0))` with :math:`t_0` being the time interval
            start.
    """
    def __init__(self, *args, **kwargs):
        super(IInitialValueProblem, self).__init__(args, kwargs)
        if "initial_value" in kwargs:
            self._initial_value = kwargs["initial_value"]
        else:
            self._initial_value = None

    @property
    def initial_value(self):
        """
        Summary
        -------
        Accessor for the initial value.

        Parameters
        ----------
        initial value : float | numpy.ndarray
            Initial value of the solution.

        Returns
        -------
        initial value : float | numpy.ndarray
            Initial value of the solution.
        """
        return self._initial_value

    @initial_value.setter
    def initial_value(self, initial_value):
        self._initial_value = initial_value

    def __str__(self):
        str = super(IInitialValueProblem, self).__str__()
        str += r", u({:.2f})={:.2f}".format(self.time_start, self.initial_value)
        return str
