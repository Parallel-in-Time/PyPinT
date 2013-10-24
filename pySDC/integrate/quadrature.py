# coding=utf-8
class Quadrature:
    """
    Provides interface for quadrature.
    """

    def __init__(self):
        """
        """
        raise NotImplementedError("Should be implemented by derivation.")

    @staticmethod
    def integrate():
        """
        Interface for integration
        """
        raise NotImplementedError("Should be implemented by specific scheme.")
