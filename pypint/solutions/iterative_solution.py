# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from .i_solution import ISolution
import numpy as np
from pypint.utilities import func_name
from pypint import LOG


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
    def __init__(self):
        super(IterativeSolution, self).__init__()
        # add one element to enable 1-based indices
        self._data = np.zeros(1, dtype=np.ndarray)
        self._errors = np.zeros(1, dtype=np.ndarray)
        self._residuals = np.zeros(1, dtype=np.ndarray)
        # make the first element a None value
        self._data[0] = None
        self._errors[0] = None
        self._residuals[0] = None
        self._used_iterations = 0

    def add_solution(self, data, **kwargs):
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
        super(IterativeSolution, self).add_solution(data, kwargs)
        if "iteration" not in kwargs:
            kwargs["iteration"] = -1
        iteration = int(kwargs["iteration"])
        _old_size = self._data.size
        # get True for each empty entry
        _empty_data_mask = np.ma.masked_equal(self._data, None).mask
        _empty_errors_mask = np.ma.masked_equal(self._errors, None).mask
        _empty_residuals_mask = np.ma.masked_equal(self._residuals, None).mask

        # resize data to fit specified iteration
        if iteration == -1 or iteration >= _old_size:
            if iteration == -1:
                _resize = _old_size + 1
            else:
                _resize = iteration + 1
            # create new index at the end of the data
            self._data = np.resize(self._data, _resize)
            self._errors = np.resize(self._errors, _resize)
            self._residuals = np.resize(self._residuals, _resize)
            # and set newly created value to None
            self._data[iteration] = None
            self._errors[iteration] = None
            self._residuals[iteration] = None

        if iteration != -1 and self._data[iteration] is not None:
            raise ValueError(func_name(self) +
                             "Data for iteration {:d} is already present. Not overriding."
                             .format(iteration))

        # fill in non-set iterations
        _empty_data_mask = np.concatenate((_empty_data_mask, [True] * (self._data.size - _old_size)))
        _empty_errors_mask = np.concatenate((_empty_errors_mask, [True] * (self._data.size - _old_size)))
        _empty_residuals_mask = np.concatenate((_empty_residuals_mask, [True] * (self._data.size - _old_size)))
        self._data[_empty_data_mask] = None
        self._errors[_empty_errors_mask] = None
        self._residuals[_empty_residuals_mask] = None

        self._data[iteration] = np.array(data, dtype=np.float64)
        if "error" in kwargs:
            self._errors[iteration] = np.array(kwargs["error"], dtype=np.float64)
        if "residual" in kwargs:
            self._residuals[iteration] = np.array(kwargs["residual"], dtype=np.float64)
        self._used_iterations += 1

    def solution(self, **kwargs):
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
        super(IterativeSolution, self).solution(kwargs)
        if "iteration" not in kwargs:
            iteration = -1
        else:
            iteration = kwargs["iteration"]

        if iteration != -1 and iteration > self._data.size:
            raise ValueError(func_name(self) +
                             "Desired iteration is not available: {:d}".format(iteration))

        return self._data[iteration]

    def error(self, **kwargs):
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
        super(IterativeSolution, self).error(kwargs)
        if "iteration" not in kwargs:
            iteration = -1
        else:
            iteration = kwargs["iteration"]

        if iteration != -1 and iteration > self._errors.size:
            raise ValueError(func_name(self) +
                             "Desired iteration is not available: {:d}".format(iteration))

        return self._errors[iteration]

    def residual(self, **kwargs):
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
        super(IterativeSolution, self).residual(kwargs)
        if "iteration" not in kwargs:
            iteration = -1
        else:
            iteration = kwargs["iteration"]

        if iteration != -1 and iteration > self._residuals.size:
            raise ValueError(func_name(self) +
                             "Desired iteration is not available: {:d}".format(iteration))

        return self._residuals[iteration]

    @property
    def data(self):
        """
        Returns
        -------
        Raw solution data with 0-based index.
        """
        return self._data[1:]

    @property
    def errors(self):
        return self._errors[1:]

    @property
    def residuals(self):
        return self._residuals[1:]

    def __str__(self):
        str = "Iterative Solution with {:d} iterations and reduction of {:.2e}:"\
              .format(self.used_iterations, self.reduction)
        for iter in range(1, self.used_iterations):
            str += "\n  Iteration {:d}: {:s}".format(iter+1, self.solution(iter))
        return str
