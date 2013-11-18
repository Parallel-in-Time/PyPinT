# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from .i_weight_function import IWeightFunction
import numpy as np
from pypint.utilities import *


class PolynomialWeightFunction(IWeightFunction):
    """
    Summary
    -------
    Provider for polynomial weight functions.

    Extended Summary
    ----------------
    Computes weights of given nodes based on a polynomial weight function of
    the form :math:`\sum_{i=0}^\infty c_i x^i`.
    By default, all powers have a coefficient of zero.

    Examples
    --------
        >>> import numpy
        >>> nodes = numpy.array([-1.0, 0.0, 1.0])
        >>> # To compute the integration weights for a given set of nodes based
        >>> # on the constant weight function 1.0 use:
        >>> # create an instance
        >>> polyWeights = PolynomialWeightFunction()
        >>> # set the coefficients of the polynom
        >>> polyWeights.init([1.0])
        >>> # compute the weights
        >>> polyWeights.evaluate(nodes)
        >>> # access the weights
        >>> polyWeights.weights
    """
    def __init__(self):
        super(self.__class__, self).__init__()
        self._coefficients = np.zeros(0)

    def init(self, coeffs, func=None):
        """
        Summary
        -------
        Sets and defines the weights function.

        Parameters
        ----------
        coeffs : numpy.ndarray|list
            Array of coefficients of the polynomial.

        func : string of format ``c0 + c1 x^1 + c2 x^2...``
            String representation of the polynomial.

        Notes
        -----
        Parsing of a string representation of the polynomial is not yet
        implemented.
        Usage will lead to a `NotImplementedError` exception.
        """
        super(self.__class__, self).init(coeffs, func)
        if func is not None and isinstance(func, str):
            # TODO: implement parsing of polynomial function string
            raise NotImplementedError(func_name(self) +
                                      "Parsing of polynomial function as string not yet possible.")
        elif coeffs is not None and \
                (isinstance(coeffs, np.ndarray) or isinstance(coeffs, list)):
            self.coefficients = np.array(coeffs)

    def evaluate(self, nodes):
        super(self.__class__, self).evaluate(nodes)
        raise NotImplementedError(func_name(self) +
                                  "Not yet implemented.")

    def add_coefficient(self, coefficient, power):
        """
        Summary
        -------
        Adds or sets the coefficient :math:`c` of :math:`cx^p` for a specific
        :math:`p`.

        Extended Summary
        ----------------
        The polynomial gets automatically extended to hold the new coefficient
        in case it didn't included the specified power previously.
        Unset, but skipped powers have a coefficient of zero by default.

        Parameters
        ----------
        coefficient : float
            Coefficient :math:`c` of :math:`cx^p`.

        power : integer
             Power :math:`p` of :math:`cx^p`.

        Examples
        --------
        >>> polyWeights = PolynomialWeightFunction()
        >>> # To set the coefficient of x^3 to 3.14 use:
        >>> polyWeights.add_coefficient(3.14, 3)
        >>> # Similar, to set the constant coefficient 42, e.i. 42*x^0, use:
        >>> polyWeights.add_coefficient(42, 0)
        """
        if not isinstance(power, int) or power < 0:
            raise ValueError(func_name(self) +
                             "Given power ({}) is not an integer or is negative"
                             .format(power))

        if self._coefficients.size <= power+1:
            self._coefficients = np.resize(self._coefficients, (power+1))

        self._coefficients[power] = coefficient

    @property
    def coefficients(self):
        """
        Summary
        -------
        Accessor for the polynomial's coefficients.

        Extended Summary
        ----------------
        To add or alter single coefficients, see :py:meth:`.add_coefficient`.

        Returns
        -------
        coefficients : numpy.ndarray
            Coefficients :math:`c_i` of the polynomial
            :math:`\sum_{i=0}^\infty c_i x^i`.

        Parameters
        ----------
        coefficients : numpy.ndarray
            Coefficients of the polynomial.

        Raises
        ------
        ValueError
            If ``coefficients`` is not a ``numpy.ndarray`` *(only Setter)*.
        """
        return self._coefficients

    @coefficients.setter
    def coefficients(self, coefficients):
        if isinstance(coefficients, np.ndarray):
            self._coefficients = coefficients
        else:
            raise ValueError(func_name(self) +
                             "Coefficients need to be a numpy.ndarray")
