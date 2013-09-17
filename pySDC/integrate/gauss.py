import numpy as np
from scipy import linalg
import pySDC.settings as config
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
        """
        _a = begin
        _b = end

        if _a == _b or (_b - _a) <= 0.0:
            raise ValueError("Integration interval must be non-zero positive (end - begin = " + str(_b - _a) + ").")

        _trans = Gauss.transform(_a, _b)
        _nw = {'nodes': [], 'weights': []}
        if type == "legendre":
            _nw = Gauss.gauss_legendre_nodes_and_weights(nPoints)
        else:
            _nw = { 'nodes': Gauss.nodes(nPoints), 'weights': Gauss.weights(nPoints) }

        _result = {'full': 0.0, 'partial': 0.0}
        for i in range(0, len(_nw['nodes'])):
            _result['full'] += _nw['weights'][i] * func(t, _trans[0] * _nw['nodes'][i] + _trans[1])
            if i >= lower and i <= upper:
                _result['partial'] += _nw['weights'][i] * func(t, _trans[0] * _nw['nodes'][i] + _trans[1])
        _result['full'] *= _trans[0]
        _result['partial'] *= _trans[0]

        if lower and upper:
#             print("Gauss.integrate() >> [" + str(_trans[0] * _nodes[lower] + _trans[1]) + "," + str(_trans[0] * _nodes[upper] + _trans[1]) + "] in [" + str(begin) + "," + str(end) + "]")
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
        """
        Gauss-Lobatto nodes for 3 to 5 integration points
        """
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
        """
        Gauss-Lobatto weights for 3 to 5 integration points
        """
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
    def gauss_legendre_nodes_and_weights(nPoints):
        """
        """
        nPoints = float(nPoints)

        if nPoints < 2:
            raise ValueError("Gauss-Legendre quadrature does not work with less than three points.")

        # Building the companion matrix CM
        # CM is such that det(xI-CM)=P_n(x), with P_n the Legendre polynomial
        # under consideration. Moreover, CM will be constructed in such a way
        # that it is symmetrical.
        j = np.linspace(start=1, stop=nPoints - 1, num=nPoints - 1)
#         print("j:\n"+str(j))
        a = j / np.sqrt(4.0 * j ** 2 - 1.0)
#         print("a:\n"+str(a))
        CM = np.diag(a, 1) + np.diag(a, -1)
#         print("CM:\n"+str(CM))

        # Determining the abscissas (x) and weights (w)
        # - since det(xI-CM)=P_n(x), the abscissas are the roots of the
        #   characteristic polynomial, i.d. the eigenvalues of CM;
        # - the weights can be derived from the corresponding eigenvectors.
        [L, V] = linalg.eig(CM)
#         print("V:\n"+str(V))
#         print("L:\n"+str(L))
        ind = np.argsort(L)
#         print("ind:\n"+str(ind))
        x = L[ind]
#         print("x:\n"+str(x))
        V = V[:, ind].transpose()
#         print("V':\n"+str(V))
#         print("V[0,:]:\n"+str(V[0,:]))
#         print("V[:,0]:\n"+str(V[:,0]))
        w = 2.0 * np.asarray(V[:,0]) ** 2.0
#         print("w:\n"+str(w))

        return {'nodes': np.around(x.real, config.PRECISION), 
                'weights': np.around(w.real, config.PRECISION)}

    @staticmethod
    def transform(a, b):
        return [(b - a) / 2.0, (b + a) / 2.0]
