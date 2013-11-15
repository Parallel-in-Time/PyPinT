# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""


class ILevelTransitionProvider(object):
    def __init__(self):
        self.__prolongation_operator = None
        self.__restringation_operator = None

    def prolongate(self, coarse_data):
        pass

    def restringate(self, fine_data):
        pass

    @property
    def prolongation_operator(self):
        return self.__prolongation_operator

    @prolongation_operator.setter
    def prolongation_operator(self, prolongation_operator):
        self.__prolongation_operator = prolongation_operator

    @property
    def restringation_operator(self):
        return self.__restringation_operator

    @restringation_operator.setter
    def restringation_operator(self, restringation_operator):
        self.__restringation_operator = restringation_operator
