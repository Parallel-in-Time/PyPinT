import numpy as np
from pySDC.integrate.gauss import Gauss


class SDC(object):
    """
    """

    def __init__(self):
        """
        """
        self.fnc = lambda t, x:-1.0 * float(x)
        self.initial_value = 1.0
        self.timeRange = [0.1, 1.0]
        self.timeSteps = 10
        self.numSubsteps = 5
        self.iterations = 5
        self.__sol = np.zeros((self.iterations, self.timeSteps, self.numSubsteps), dtype=float)

    def solve(self):
        """
        solves a given problem setup
        """

        _substeps = np.zeros((self.timeSteps, self.numSubsteps), dtype=float)
        _dt_n = self.timeRange[1] - self.timeRange[0] / self.timeSteps
        _nodes = Gauss.nodes(self.numSubsteps)

        # Set initial values and compute substep points
        for t_n_i in range(0, self.timeSteps):
            _t_n = self.timeRange[0] + t_n_i * _dt_n

            # transform [t_n, t_n+_dt_n] into Gauss nodes
            _trans = Gauss.transform(_t_n, _t_n + _dt_n)
            assert len(_trans) == 2, "Coordinate transformation failed"
            assert len(_substeps) > t_n_i, "Substeps not initialized (len(_substeps)=" + str(len(_substeps)) + ")"

            for step in range(0, self.numSubsteps):
                assert len(_nodes) > step, "Fever nodes than steps"
                _substeps[t_n_i][step] = _trans[0] * _nodes[step] + _trans[1]
            print("_substeps[" + str(t_n_i) + "] = " + str(_substeps[t_n_i]))

            self.__sol[0][t_n_i] = np.asarray([self.initial_value] * self.numSubsteps)
            print("__sol[0][" + str(t_n_i) + "] = " + str(self.solution[0][t_n_i]))

        # Compute SDC iterations
        for k in range(1, self.iterations):
            self.__sol[k][0][0] = self.initial_value
            for t_n_i in range(0, self.timeSteps):
                _t_n = self.timeRange[0] + t_n_i * _dt_n
                self.__sol[k][t_n_i][0] = self.__sol[k][t_n_i - 1][-1]
                for t_m_i in range(1, self.numSubsteps):
                    _t_m = _substeps[t_n_i][t_m_i]
                    _dt_m = _t_m - _substeps[t_n_i][t_m_i - 1]
                    assert _dt_m > 0.0, "dt_m should be larger 0"

                    # compute Eqn. 2.7 in explicit form
                    self.__sol[k][t_n_i][t_m_i] = self.__sol[k][t_n_i][t_m_i - 1] + \
                        _dt_m * (self.fnc(_t_m, self.__sol[k][t_n_i][t_m_i - 1]) - \
                                 self.fnc(_t_m, self.__sol[k - 1][t_n_i][t_m_i])) + \
                        Gauss.integrate(func=self.fnc, t=_t_m, begin=_t_n, end=(_t_n + _dt_n), nPoints=self.numSubsteps, lower=(t_m_i - 1), upper=t_m_i)
                    print("__sol[" + str(k) + "][" + str(t_n_i) + "][" + str(t_m_i) + "] = "
                           + str(self.__sol[k][t_n_i][t_m_i - 1]) + " + " + str(_dt_m) + " * (" + str(self.fnc(_t_m, self.__sol[k][t_n_i][t_m_i - 1]))
                           + " - " + str(self.fnc(_t_m, self.__sol[k - 1][t_n_i][t_m_i])) + ") + "
                           + "Gauss.integrate(func=" + str(self.fnc) + ", t=" + str(_t_m) + ", begin=" + str(_t_n) + ", end=" + str(_t_n + _dt_n) + ", nPoints=" + str(self.numSubsteps) + ", lower=" + str(t_m_i - 1) + ", upper=" + str(t_m_i) + ")")
                print("__sol[" + str(k) + "][" + str(t_n_i) + "] = " + str(self.solution[k][t_n_i]))

    @property
    def solution(self):
        """
        solution of the last call to :py:func:`SDC.solve`

        Returns
        -------
        list[Decimal]
        """
        return self.__sol

    @solution.deleter
    def solution(self):
        del self.__sol

    @property
    def fnc(self):
        """
        function describing the problem

        Returns
        -------
        function pointer
        """
        return self.__function

    @fnc.setter
    def fnc(self, value):
        self.__function = value

    @fnc.deleter
    def fnc(self):
        del self.__function

    @property
    def initial_value(self):
        """
        initial value of the problem

        Returns
        -------
        decimal.Decimal
        """
        return self.__initial_value

    @initial_value.setter
    def initial_value(self, value):
        self.__initial_value = value

    @initial_value.deleter
    def initial_value(self):
        del self.__initial_value

    @property
    def time_range(self):
        """
        pair of start and end time

        Returns
        -------
        list[decimal.Decimal]

        Raises
        ------
        ValueError
            On setting if time range is non-positive or zero
        """
        return self.__timeRange

    @time_range.setter
    def time_range(self, value):
        if value[1] <= value[0]:
            raise ValueError("Time interval must be non-zero positive [start, "
                             "end]: [" + str(value[0]) + ", " + str(value[1])
                             + "]")
        self.__timeRange = [value[0], value[1]]

    @time_range.deleter
    def time_range(self):
        del self.__timeRange

    @property
    def time_steps(self):
        """
        number of time steps

        Returns
        -------
        int

        Raises
        ------
        ValueError
            on setting if number steps is not positive
        """
        return self.__timeSteps

    @time_steps.setter
    def time_steps(self, value):
        if value <= 0:
            raise ValueError("At least one time step is neccessary.")
        self.__timeSteps = value

    @time_steps.deleter
    def time_steps(self):
        del self.__timeSteps

    @property
    def num_substeps(self):
        """
        number of substeps of each time step

        Returns
        -------
        int

        Raises
        ------
        ValueError
            on setting if number substeps is not positive
        """
        return self.__numSubsteps

    @num_substeps.setter
    def num_substeps(self, value):
        if value <= 0:
            raise ValueError("At least one substep is neccessary.")
        self.__numSubsteps = value

    @num_substeps.deleter
    def num_substeps(self):
        del self.__numSubsteps

    @property
    def iterations(self):
        """
        number if SDC iterations

        Returns
        -------
        int

        Raises
        ------
        ValueError
            on setting if iterations is not positive
        """
        return self.__iterations

    @iterations.setter
    def iterations(self, value):
        if value <= 0:
            raise ValueError("At least one iteration is neccessary.")
        self.__iterations = value

    @iterations.deleter
    def iterations(self):
        del self.__iterations
