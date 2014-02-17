# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
import warnings

from pypint.solutions.i_solution import ISolution
from pypint.solutions.data_storage.step_solution_data import StepSolutionData


class FinalSolution(ISolution):
    """Storage for the final solution of a solver.

    The storage data type is defined as :py:class:`.StepSolutionData`.
    """

    def __init__(self, *args, **kwargs):
        super(FinalSolution, self).__init__(*args, **kwargs)
        # As this solution only stores the very last step, `StepSolutionData` is the solution data type.
        self._data_type = StepSolutionData

    def add_solution(self, *args, **kwargs):
        """Sets and resets the stored solution data storage.

        This method is constructing a new :py:class:`.StepSolutionData` object from the given list of
        arguments.

        Raises
        ------
        UserWarning :
            If there is already a :py:class:`.StepSolutionData` object stored.

        See Also
        --------
        .StepSolutionData :
            for available and valid parameters.
        """
        if self._data:
            warnings.warn("There is already a solution data object stored. Overriding it.")
        self._data = self._data_type(*args, **kwargs)

    @property
    def value(self):
        """Proxies :py:attr:`.StepSolutionData.value`
        """
        return self._data.value if self._data else None

    @property
    def time_point(self):
        """Proxies :py:attr:`.StepSolutionData.time_point`
        """
        return self._data.time_point if self._data else None

    @property
    def error(self):
        """Proxies :py:attr:`.StepSolutionData.error`
        """
        return self._data.error if self._data else None

    @property
    def residual(self):
        """Proxies :py:attr:`.StepSolutionData.residual`
        """
        return self._data.residual if self._data else None


__all__ = ['FinalSolution']
