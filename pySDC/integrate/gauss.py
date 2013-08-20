from decimal import Decimal
from pySDC.integrate.quadrature import Quadrature

class Gauss(Quadrature):
    """
    """

    def __init__(self):
        """
        """

    @staticmethod
    def integrate(func=lambda x: Decimal(1.0), begin=0, end=1, nPoints=3):
        """
        """
        _a = Decimal(begin)
        _b = Decimal(end)

        if _a == _b or (_b - _a) <= Decimal(0.0):
            raise ValueError("Integration interval must be non-zero positive (end - begin = " + str(_b - _a) + ").")

        _trans = Gauss.transform(_a, _b)
        _nodes = Gauss.nodes(nPoints)
        _weights = Gauss.weights(nPoints)
        _result = Decimal(0.0)
        for i in range(0, len(_nodes)):
            _result += _weights[i] * Decimal(func(_trans[0] * _nodes[i] + _trans[1]))
        _result *= _trans[0]
        return _result

    @staticmethod
    def partial_integrate(lower, upper, values):
        _result = Decimal(0.0)
        _weights = Gauss.weights(len(values))
        for i in range(lower, upper + 1):
            _result += _weights[i] * values[i]
        return _result

    @staticmethod
    def nodes(nPoints):
        if nPoints == 3:
            return [ Decimal(-1.0),
                     Decimal(0.0),
                     Decimal(1.0) ]
        elif nPoints == 4:
            return [ Decimal(-1.0),
                     Decimal(-1.0) / Decimal(5) * Decimal(5).sqrt(),
                     Decimal(1.0) / Decimal(5) * Decimal(5).sqrt(),
                     Decimal(1.0) ]
        elif nPoints == 5:
            return [ Decimal(-1.0),
                     Decimal(-1.0) / Decimal(7) * Decimal(21).sqrt(),
                     Decimal(0.0),
                     Decimal(1.0) / Decimal(7) * Decimal(21).sqrt(),
                     Decimal(1.0) ]
        elif nPoints < 3:
            raise ValueError("Gauss-Lobatto quadrature does not work with less than three points.")
        else:
            raise NotImplementedError()

    @staticmethod
    def weights(nPoints):
        if nPoints == 3:
            return [ Decimal(1 / Decimal(3)),
                     Decimal(4 / Decimal(3)),
                     Decimal(1 / Decimal(3)) ]
        elif nPoints == 4:
            return [ Decimal(1 / Decimal(6)),
                     Decimal(5) / Decimal(6),
                     Decimal(5) / Decimal(6),
                     Decimal(1 / Decimal(6)) ]
        elif nPoints == 5:
            return [ Decimal(1) / Decimal(10),
                     Decimal(49) / Decimal(90),
                     Decimal(32) / Decimal(45),
                     Decimal(49) / Decimal(90),
                     Decimal(1) / Decimal(10) ]
        elif nPoints < 3:
            raise ValueError("Gauss-Lobatto quadrature does not work with less than three points.")
        else:
            raise NotImplementedError()

    @staticmethod
    def transform(a, b):
        return [Decimal((b - a) / Decimal(2)), Decimal((b + a) / Decimal(2))]
