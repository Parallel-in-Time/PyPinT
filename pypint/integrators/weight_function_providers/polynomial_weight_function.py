# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from pypint.integrators.weight_function_providers.i_weight_function import IWeightFunction
import numpy as np


class PolynomialWeightFunction(IWeightFunction):
    """
    provider for polynomial weight functions

    Computes weights of given nodes based on a polynomial weight function of
    the form :math:`\sum_{i=0}^\infty c_i x^i`
    """
    def __init__(self):
        super().__init__()
        self._coefficients = np.zeros(0)

    def init(self, coeffs, func=None):
        """
        sets and defines the weights function

        :param coeffs: array of coefficients of polynomial
        :type coeffs:  numpy.ndarray | list
        :param func:   string representation of the polynomial
        :type func:    string of format ``c0 + c1 x^1 + c2 x^2...``
        """
        if func is not None and isinstance(func, str):
            # TODO: implement parsing of polynomial function string
            raise NotImplementedError(self.__qualname__ + ".init(): " +
                                      "Parsing of polynomial function as string not yet possible.")
        elif coeffs is not None and \
                (isinstance(coeffs, np.ndarray) or isinstance(coeffs, list)):
            self.coefficients = np.array(coeffs)

    def evaluate(self, nodes):
        raise NotImplementedError(self.__qualname__ + ".evaluate(): " +
                                  "Not yet implemented.")

    def add_coefficient(self, coefficient, potent):
        """
        adds or sets the coefficient for :math:`x^p`

        To set the coefficient of :math:`x^3` to ``3.14`` use::
            add_coefficient(3.14, 3)
        Similar, to set the constant coefficient ``42``, e.g. :math:`x^0`, use::
            add_coefficient(42, 0)

        :param coefficient: coefficient of :math:`x^p`
        :type coefficient:  float
        :param potent:      potency of :math:`x` this is the coefficient for
        :type potent:       integer
        """
        if not isinstance(potent, int) or potent < 0:
            raise ValueError(self.__qualname__ + ".add_coefficient(): " +
                             "Given potent ({}) is not an integer or negative"
                             .format(potent))

        if self._coefficients.size <= potent+1:
            self._coefficients = np.resize(self._coefficients, (potent+1))

        self._coefficients[potent] = coefficient

    @property
    def coefficients(self):
        """
        accessor for the polynomial's coefficients

        To add or alter single coefficients, see :py:func:`.add_coefficient`.

        **Getter**

        :returns: coefficients of the polynomial
        :rtype:   numpy.ndarray


        **Setter**

        :param coefficients: coefficients of the polynomial
        :type coefficients:  numpy.ndarray
        :raises: **ValueError** if ``coefficients`` is not a ``numpy.ndarray``
        """
        return self._coefficients

    @coefficients.setter
    def coefficients(self, coefficients):
        if isinstance(coefficients, np.ndarray):
            self._coefficients = coefficients
        else:
            raise ValueError(self.__qualname__ + ".coefficients(): "
                             "Coefficients need to be a numpy.ndarray")
