import numpy as np
from pySDC.integrate.quadrature import Quadrature

class Gauss(Quadrature):
    """
    """

    def __init__(self):
        """
        """

    @staticmethod
    def integrate(func=lambda t, x: 1.0, begin=0, end=1, nPoints=3, t=1.0, lower=None, upper=None):
        """
        """
        _a = begin
        _b = end

        if _a == _b or (_b - _a) <= 0.0:
            raise ValueError("Integration interval must be non-zero positive (end - begin = " + str(_b - _a) + ").")

        _trans = Gauss.transform(_a, _b)
        _nodes = Gauss.nodes(nPoints)
        _weights = Gauss.weights(nPoints)
        _result = {'full': 0.0, 'partial': 0.0}
        for i in range(0, len(_nodes)):
            _result['full'] += _weights[i] * func(t, _trans[0] * _nodes[i] + _trans[1])
            if i >= lower and i <= upper:
                _result['partial'] += _weights[i] * func(t, _trans[0] * _nodes[i] + _trans[1])
        _result['full'] *= _trans[0]
        _result['partial'] *= _trans[0]

        if lower and upper:
            print("Gauss.integrate() >> [" + str(_trans[0] * _nodes[lower] + _trans[1]) + "," + str(_trans[0] * _nodes[upper] + _trans[1]) + "] in [" + str(begin) + "," + str(end) + "]")
            return _result['partial']
        else:
            return _result['full']

    @staticmethod
    def partial_integrate(lower, upper, values):
        _result = 0.0
        _weights = Gauss.weights(len(values))
        for i in range(lower, upper + 1):
            _result += _weights[i] * values[i]
        return _result

    @staticmethod
    def nodes(nPoints):
        if nPoints == 3:
            return [ -1.0,
                     0.0,
                     1.0 ]
        elif nPoints == 4:
            return [ -1.0,
                     - 1.0 / 5.0 * np.sqrt(5),
                     1.0 / 5.0 * np.sqrt(5),
                     1.0 ]
        elif nPoints == 5:
            return [ -1.0,
                     - 1.0 / 7.0 * np.sqrt(21),
                     0.0,
                     1.0 / 7.0 * np.sqrt(21),
                     1.0 ]
        elif nPoints < 3:
            raise ValueError("Gauss-Lobatto quadrature does not work with less than three points.")
        else:
            raise NotImplementedError()

    @staticmethod
    def weights(nPoints):
        if nPoints == 3:
            return [ 1.0 / 3.0,
                     4.0 / 3.0,
                     1.0 / 3.0 ]
        elif nPoints == 4:
            return [ 1.0 / 6.0,
                     5.0 / 6.0,
                     5.0 / 6.0,
                     1.0 / 6.0 ]
        elif nPoints == 5:
            return [ 1.0 / 10.0,
                     49.0 / 90.0,
                     32.0 / 45.0,
                     49.0 / 90.0,
                     1.0 / 10.0 ]
        elif nPoints < 3:
            raise ValueError("Gauss-Lobatto quadrature does not work with less than three points.")
        else:
            raise NotImplementedError()

    @staticmethod
    def transform(a, b):
        return [(b - a) / 2.0, (b + a) / 2.0]
