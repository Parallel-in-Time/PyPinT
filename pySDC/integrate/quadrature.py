"""
Quadrature Interface
"""


class Quadrature(object):
    """
    Provides interface for quadrature.
    """

    def __init__(self):
        """
        Initialization
        """
        raise NotImplementedError("Should be implemented by derivation.")

    @staticmethod
    def integrate():
        """
        Interface for integration
        """
        raise NotImplementedError("Should be implemented by specific scheme.")
