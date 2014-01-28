# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from .i_solution import ISolution
from .step_solution_data import StepSolutionData
import warnings


class FinalSolution(ISolution):
    """
    Summary
    -------
    Storage for the final solution of a solver.
    """

    def __init__(self, *args, **kwargs):
        # As this solution only stores the very last step, `StepSolutionData` is the solution data type.
        super(FinalSolution, self).__init__(*args, solution_data_type=StepSolutionData, **kwargs)

    def add_solution(self, *args, **kwargs):
        """
        Summary
        -------
        Sets and resets the stored solution data storage.

        Extended Summary
        ----------------
        This method is constructing a new :py:class:`.solutions.StepSolutionData` object from the given list of
        arguments.

        Raises
        ------
        UserWarning :
            If there is already a :py:class:`.solutions.StepSolutionData` object stored.

        See Also
        --------
        .solutions.StepSolutionData : for available and valid parameters.
        """
        if self._data:
            warnings.warn("There is already a solution data object stored. Overriding it.")
        self._data = self._data_type(*args, **kwargs)

    @property
    def value(self):
        """
        Summary
        -------
        Proxies :py:attr:`.solutions.StepSolutionData.value`
        """
        return self._data.value if self._data else None

    @property
    def time_point(self):
        """
        Summary
        -------
        Proxies :py:attr:`.solutions.StepSolutionData.time_point`
        """
        return self._data.time_point if self._data else None

    @property
    def error(self):
        """
        Summary
        -------
        Proxies :py:attr:`.solutions.StepSolutionData.error`
        """
        return self._data.error if self._data else None

    @property
    def residual(self):
        """
        Summary
        -------
        Proxies :py:attr:`.solutions.StepSolutionData.residual`
        """
        return self._data.residual if self._data else None


__all__ = ['FinalSolution']
