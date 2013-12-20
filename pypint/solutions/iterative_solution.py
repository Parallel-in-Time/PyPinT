# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from .i_solution import ISolution
import numpy as np
from pypint.utilities import assert_condition


class IterativeSolution(ISolution):
    """
    Summary
    -------
    Storage for the solutions of an iterative solver.

    Extended Summary
    ----------------
    A new solution of a specific iteration can be added via
    :py:func:`.add_solution` and queried via :py:func:`.solution`.
    """

    def __init__(self, numeric_type=np.float):
        super(IterativeSolution, self).__init__(numeric_type)
        # add one element to enable 1-based indices
        self._points = np.zeros(0, dtype=np.float)
        self._data = []
        self._used_iterations = 0

    def add_solution(self, points, values, *args, **kwargs):
        """
        Summary
        -------
        Adds a new solution of the specified iteration.

        Extended Summary
        ----------------
        A copy of the given data is stored as a ``numpy.float64`` array at the given iteration
        index.

        Parameters
        ----------
        data : numpy.ndarray
             solution data

        ``iteration`` : integer
            Index of the iteration of this solution (1-based).
            ``-1`` auto-appends the solution.

        Raises
        ------
        ValueError
            * if ``iteration`` is not given
            * if there are more than ``iteration`` solutions already stored

        See Also
        --------
        .ISolution.add_solution
            overridden method; see for further named arguments
        """
        super(IterativeSolution, self).add_solution(points, values, *args, **kwargs)

        assert_condition(np.all(self._points == points), ValueError, "Given points are not equal stored ones.", self)

        # get iteration index (1-based)
        iteration = int(kwargs["iteration"]) if "iteration" in kwargs else -1

        # given iteration index of -1 means appending as last
        if iteration == -1:
            iteration = len(self._data) + 1  # (+1 because of 1-based index)

        # convert iteration index to be 0-based
        iteration -= 1

        # NOTE: from here on, iteration is 0-based

        # will we insert or append (incl. possible skipping)?
        _append = (iteration >= len(self._data))

        # we do not allow to override existing solutions (non-None elements)
        if not _append:
            assert_condition(self._data[iteration] is None,
                            ValueError, "Data for iteration {:d} is already present. Not overriding."
                                        .format(iteration + 1), self)

        # if not simple append, fill in skipped iterations
        while _append and len(self._data) < iteration:
            self._data.append(None)

        _append = (iteration == len(self._data))

        # prepare values
        _values = values.astype(dtype=self.numeric_type)
        _errors = kwargs["error"].astype(np.float) if "error" in kwargs else None
        _residuals = kwargs["residual"].astype(np.float) if "residual" in kwargs else None

        # prepare data object
        if _append:
            self._data.append(ISolution.IterationData())
        else:
            self._data[iteration] = ISolution.IterationData()

        # fill data object with values
        self._data[iteration].init(iteration=iteration, values=_values, errors=_errors, residuals=_residuals,
                                   numeric_type=self.numeric_type)
        self._used_iterations += 1

    def solution(self, *args, **kwargs):
        """
        Summary
        -------
        Queries the solution vector of the given iteration.

        Parameters
        ----------
        iteration : integer
            Index of the desired solution vector (1-based).
            Defaults to -1.
            Index ``1`` is the first solution, ``-1`` the last and final solution vector.

        Returns
        -------
        solution vector : numpy.ndarray
            Solution of the given iteration.

        Raises
        ------
        ValueError
            If ``iteration`` is not available.

        See Also
        --------
        .ISolution.solution
            overridden method
        """
        iteration = kwargs["iteration"] if "iteration" in kwargs else -1
        if iteration == -1:
            iteration = len(self._data)

        assert_condition(iteration <= len(self._data),
                        ValueError, "Desired iteration is not available: {:d}".format(iteration), self)

        return self._data[iteration - 1].values if isinstance(self._data[iteration - 1], ISolution.IterationData) \
            else None

    def error(self, *args, **kwargs):
        """
        Parameters
        ----------
        iteration : integer
            Index of the iteration of the desired error (1-based).
            Defaults to -1.
            Index ``1`` is the first error, ``-1`` the last and final error vector.

        Returns
        -------
        error vector : numpy.ndarray
            Error of the given iteration.

        Raises
        ------
        ValueError
            If ``iteration`` is not available.

        See Also
        --------
        .ISolution.error
            overridden method
        """
        iteration = kwargs["iteration"] if "iteration" in kwargs else -1
        if iteration == -1:
            iteration = len(self._data)

        assert_condition(iteration <= len(self._data),
                        ValueError, "Desired iteration is not available: {:d}".format(iteration), self)

        return self._data[iteration - 1].errors if isinstance(self._data[iteration - 1], ISolution.IterationData) \
            else None

    def residual(self, *args, **kwargs):
        """
        Parameters
        ----------
        iteration : integer
            Index of the iteration of the desired residual (1-based).
            Defaults to -1.
            Index ``1`` is the first residual, ``-1`` the last and final residual vector.

        Returns
        -------
        residual vector : numpy.ndarray
            Residual of the given iteration.

        Raises
        ------
        ValueError
            If ``iteration`` is not available.

        See Also
        --------
        .ISolution.residual
            overridden method
        """
        iteration = kwargs["iteration"] if "iteration" in kwargs else -1
        if iteration == -1:
            iteration = len(self._data)

        assert_condition(iteration <= len(self._data),
                        ValueError, "Desired iteration is not available: {:d}".format(iteration), self)

        return self._data[iteration - 1].residuals if isinstance(self._data[iteration - 1], ISolution.IterationData) \
            else None

    @property
    def values(self):
        """
        Returns
        -------
        Raw solution data with 0-based index.
        """
        return [i.values if isinstance(i, ISolution.IterationData) else None for i in self._data]

    @property
    def errors(self):
        return [i.errors if isinstance(i, ISolution.IterationData) else None for i in self._data]

    @property
    def residuals(self):
        return [i.residuals if isinstance(i, ISolution.IterationData) else None for i in self._data]

    def __str__(self):
        out = "Iterative Solution with {:d} iterations and reduction of {:.2e}:"\
              .format(self.used_iterations, self.reductions["solution"][-1])
        for i in range(1, self.used_iterations):
            out += "\n  Iteration {:d}: {:s}".format(i + 1, self.solution(iteration=i))
        return out
