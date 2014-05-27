# coding=utf-8
"""Some mathematical utility functions

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""


def lagrange_polynome(j, base_points, x):
    """Evaluates :math:`j`-th Lagrange polynomial based on ``base_points`` at :math:`x`

    For a given set of :math:`n` nodes :math:`\\vec{b}` (``base_points``) the :math:`j`-th Lagrange polynomial is
    constructed and evaluated at the given point :math:`x`.

    .. math::

        P_j(x) = \\prod_{m=1, m \\neq j}^{n} \\frac{x - b_m}{b_j - b_m}

    Parameters
    ----------
    j : :py:class:`int`
        descriptor of the Lagrange polynomial

    base_points : :py:class:`numpy.ndarray` of :math:`n` :py:class:`float`
        points to construct the Lagrange polynome on

    x : :py:class:`float`
        point to evaluate the Lagrange polynome at

    Returns
    -------
    value : :py:class:`float`
        value of the specified Lagrange polynome
    """
    _val = 1.0
    for m in range(0, base_points.size):
        if m != j:
            _val *= (x - base_points[m]) / (base_points[j] - base_points[m])
    return _val
