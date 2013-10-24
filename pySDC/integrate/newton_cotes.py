"""
Newton-Cotes Quadrature
"""

from decimal import Decimal
from pySDC.integrate.quadrature import Quadrature


class NewtonCotes(Quadrature):
    """
    Provides integration with Newton-Codes quadrature.
    """

    def __init__(self):
        """
        """
        super(NewtonCotes, self).__init__()

    @staticmethod
    def integrate(func=lambda x: 1, begin=Decimal(0), end=Decimal(1), steps=10,
                  order=1):
        """
        Integrates given function in `[begin, end]` using `nPoints` with
        Newton-Cotes-Quadrature

        :param func:  function to be integrated; requires point `x` as only
                      argument; default: constant 1 function
        :type func:   function pointer or lambda
        :param begin: start point of interval
        :type begin:  Integer or Float
        :param end:   end point of interval
        :type end:    Integer or Float
        :param steps: number of steps
        :type steps:  Integer
        :param order: number of intergration points per step
        :type order:  Integer

        :rtype: decimal.Decimal

        :raises: ValueError (if zero-interval or wrong orientation or
                 `steps`<1),
                 NotImplementedError (if `order`>4)
        """
        a = Decimal(begin)
        b = Decimal(end)

        if a == b or (b - a) <= Decimal(0.0):
            raise ValueError("Integration interval must be non-zero " +
                             "(end - begin = {:d}).".format(b - a))
        if steps < 1:
            raise ValueError("At least one step makes sense (steps={:d})."
                             .format(steps))

        step_width = (b - a) / Decimal(steps)
        result = Decimal(0.0)

        if order == 1:
            # Midpoint rule
            for i in range(0, steps):
                result += step_width * Decimal(func(a + Decimal(i + 0.5) *
                                                    step_width))
        elif order == 2:
            # Trapezoid rule
            for i in range(0, steps):
                result += step_width * Decimal(
                    func(a + i * step_width)
                    + func(a + (i + 1) * step_width)
                ) / Decimal(2)
        elif order == 3:
            # Simpson rule
            for i in range(0, steps):
                result += step_width * Decimal(
                    func(a + i * step_width)
                    + 4 * func(a + Decimal(i + 0.5) * step_width)
                    + func(a + Decimal(i + 1) * step_width)
                ) / Decimal(6)
        elif order == 4:
            # Simpson 3/8 rule
            for i in range(0, steps):
                result += step_width * Decimal(
                    func(a + i * step_width)
                    + 3 * func(a + Decimal(i + 1 / Decimal(3)) * step_width)
                    + 3 * func(a + Decimal(i + 2 / Decimal(3)) * step_width)
                    + func(a + Decimal(i + 1) * step_width)
                ) / Decimal(8)

        else:
            raise NotImplementedError("Newton-Codes integration scheme with " +
                                      "order={:d} not implemented."
                                      .format(order))

        return result
