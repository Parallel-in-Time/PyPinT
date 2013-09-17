import numpy as np
from scipy import linalg
import pySDC.globals as config
from pySDC.integrate.quadrature import Quadrature

class Gauss(Quadrature):
    """
    """

    def __init__(self):
        """
        """

    @staticmethod
    def integrate(func=lambda t, x: 1.0, begin=0, end=1, nPoints=3, t=1.0, lower=None, upper=None, type="legendre"):
        """
        integrates given function in [begin, end] using nPoints at time t with method 'type'
        """
        _a = begin
        _b = end

        if _a == _b or (_b - _a) <= 0.0:
            raise ValueError("Integration interval must be non-zero positive (end - begin = " + str(_b - _a) + ").")

        _trans = Gauss.transform(_a, _b)
        _nw = Gauss.get_nodes_and_weights(nPoints, type)

        _result = {'full': 0.0, 'partial': 0.0}
        for i in range(0, len(_nw['nodes'])):
            _result['full'] += _nw['weights'][i] * func(t, _trans[0] * _nw['nodes'][i] + _trans[1])
            if lower is not None and upper is not None and i >= lower and i <= upper:
                _result['partial'] += _nw['weights'][i] * func(t, _trans[0] * _nw['nodes'][i] + _trans[1])
        _result['full'] *= _trans[0]
        _result['partial'] *= _trans[0]

        if lower and upper:
#             print("Gauss.integrate() >> [" + str(_trans[0] * _nodes[lower] + _trans[1]) + "," + str(_trans[0] * _nodes[upper] + _trans[1]) + "] in [" + str(begin) + "," + str(end) + "]")
            return _result['partial']
        else:
            return _result['full']

    @staticmethod
    def get_nodes_and_weights(nPoints, type="legendre"):
        """
        returns integration nodes and weights for given type and number of points
        """
        if type == "legendre":
            return Gauss.legendre_nodes_and_weights(nPoints)
        elif type == "lobatto":
            return Gauss.lobatto_nodes_and_weights(nPoints)
        else:
            raise NotImplementedError("Gaus-" + str(type) + "-Quadrature not implemented/known.")

    @staticmethod
    def legendre_nodes_and_weights(nPoints):
        """
        computats nodes and weights for the Gauss-Legendre quadrature of order n>1 on [-1, +1]
        (ported from MATLAB code, reference see below)

        (original comment from MatLab source; modified)
        Unlike many publicly available functions, this function is valid for
        n>=46.
        This is due to the fact that it does not rely on MATLAB's build-in 'root'
        routines to determine the roots of the Legendre polynomial, but finds the
        roots by looking for the eigenvalues of an alternative version of the
        companion matrix of the n'th degree Legendre polynomial.
        The companion matrix is constructed as a symmetrical matrix, guaranteeing
        that all the eigenvalues (roots) will be real.
        On the contrary, MATLAB's 'roots' function uses a general form for the
        companion matrix, which becomes unstable at higher orders n, leading to
        complex roots.

        (Credit, where credit due)
        original MATLAB function by: Geert Van Damme <geert@vandamme-iliano.be> (February 21, 2010)
        """
        nPoints = float(nPoints)

        if nPoints < 2:
            raise ValueError("Gauss-Legendre quadrature does not work with less than three points.")

        # Building the companion matrix CM
        # CM is such that det(xI-CM)=P_n(x), with P_n the Legendre polynomial
        # under consideration. Moreover, CM will be constructed in such a way
        # that it is symmetrical.
        j = np.linspace(start=1, stop=nPoints - 1, num=nPoints - 1)
        a = j / np.sqrt(4.0 * j ** 2 - 1.0)
        CM = np.diag(a, 1) + np.diag(a, -1)

        # Determining the abscissas (x) and weights (w)
        # - since det(xI-CM)=P_n(x), the abscissas are the roots of the
        #   characteristic polynomial, i.d. the eigenvalues of CM;
        # - the weights can be derived from the corresponding eigenvectors.
        [L, V] = linalg.eig(CM)
        ind = np.argsort(L)
        x = L[ind]
        V = V[:, ind].transpose()
        w = 2.0 * np.asarray(V[:, 0]) ** 2.0

        return {'nodes': np.around(x.real, config.DIGITS),
                'weights': np.around(w.real, config.DIGITS)}

    @staticmethod
    def transform(a, b):
        """
        calculates transformation coefficients to map [a,b] to [-1,1]
        
        see: http://en.wikipedia.org/wiki/Gaussian_quadrature#Change_of_interval
        """
        return [(b - a) / 2.0, (b + a) / 2.0]

    @staticmethod
    def lobatto_nodes_and_weights(nPoints):
        """
        Gauss-Lobatto nodes and weights for 3 to 5 integration points (hard coded)

        source of values: http://en.wikipedia.org/wiki/Gaussian_quadrature#Gauss.E2.80.93Lobatto_rules
        """
        if nPoints == 3:
            return {'nodes': [ -1.0,
                                0.0,
                                1.0 ],
                    'weights': [ 1.0 / 3.0,
                                 4.0 / 3.0,
                                 1.0 / 3.0 ]}
        elif nPoints == 4:
            return {'nodes': [ -1.0,
                               -1.0 / 5.0 * np.sqrt(5),
                                1.0 / 5.0 * np.sqrt(5),
                                1.0 ],
                    'weights': [ 1.0 / 6.0,
                                 5.0 / 6.0,
                                 5.0 / 6.0,
                                 1.0 / 6.0 ]}
        elif nPoints == 5:
            return {'nodes': [ -1.0,
                               -1.0 / 7.0 * np.sqrt(21),
                                0.0,
                                1.0 / 7.0 * np.sqrt(21),
                                1.0 ],
                    'weights': [  1.0 / 10.0,
                                 49.0 / 90.0,
                                 32.0 / 45.0,
                                 49.0 / 90.0,
                                  1.0 / 10.0 ]}
        elif nPoints < 3:
            raise ValueError("Gauss-Lobatto quadrature does not work with less than three points.")
        else:
            raise NotImplementedError("Gauss-Lobatto with " + str(nPoints) + " is not implemented yet.")
