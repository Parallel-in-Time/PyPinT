import numpy as np
from pySDC.integrate.gauss import Gauss


class SDC(object):
    """
    """

    def __init__(self):
        """
        """
        self.fnc = lambda t, phi_t:-1.0
        self.exact = lambda t:-t + 1.0
        self.initial_value = 1.0
        self.time_range = [0.0, 1.0]
        self.time_steps = 1
        self.num_substeps = 3
        self.iterations = 3
        self.__sol = np.zeros((self.iterations + 1, self.time_steps, self.num_substeps + 2), dtype=float)
        self._substeps = np.zeros((self.time_steps, self.num_substeps + 2), dtype=float)
        self._dt_n = float(self.time_range[1] - self.time_range[0]) / float(self.time_steps)
        self._reduction = np.zeros((self.iterations, self.time_steps, self.num_substeps + 2), dtype=float)
        self._error = np.ones((self.iterations + 1, self.time_steps, self.num_substeps + 2), dtype=float)
        self.verbosity = 4

    def solve(self):
        """
        solves a given problem setup
        """
        self._substeps = np.zeros((self.time_steps, self.num_substeps + 2), dtype=float)
        self._dt_n = float(self.time_range[1] - self.time_range[0]) / float(self.time_steps)
        self._reduction = np.zeros((self.iterations, self.time_steps, self.num_substeps + 2), dtype=float)
#         print("[{: f}, {: f}], {: f} ==> {: f}".format(self.time_range[0], self.time_range[1], self.time_steps, self._dt_n))
        _nodes = Gauss.get_nodes_and_weights(self.num_substeps, "legendre")['nodes']

        # #
        # # Set initial values and compute substep points
        # #
        self._substeps[0][0] = self.time_range[0]
        self._substeps[-1][-1] = self.time_range[1]

        # iterate main time steps
        for t_n_i in range(0, self.time_steps):
            _t_n = self.time_range[0] + t_n_i * self._dt_n
#             print("t_n_i={:d} (t_n={: f}, _dt_n={: f})".format(t_n_i, _t_n, self._dt_n))

            # transform [t_n, t_n+_dt_n] into Gauss nodes
            _trans = Gauss.transform(_t_n, _t_n + self._dt_n)
            assert len(_trans) == 2, "Coordinate transformation failed (len(_trans)={:d}).".format(len(_trans))
            assert len(self._substeps) > t_n_i, "Substeps not correctly initialized (len(_substeps)={:d}).".format(len(self._substeps))

            self.__sol[0][t_n_i] = np.asarray([self.initial_value] * (self.num_substeps + 2))

            if t_n_i > 0:
                # Compute initial values for _t_n via Standard Euler
                self.__sol[0][t_n_i][0] = self.__sol[0][t_n_i - 1][0] + self._dt_n * self.fnc(_t_n)
                # copy values
                self._substeps[t_n_i][0] = self._substeps[t_n_i - 1][-1]

            # compute substep points for current time step
            for t_m_i in range(0, self.num_substeps):
#                 print("  t_m_i={:d}".format(t_m_i))
                assert len(_nodes) > t_m_i, "Fever nodes than steps"
#                 print("    _t_n={: f} (node={: f})".format(_trans[0] * _nodes[t_m_i] + _trans[1], _nodes[t_m_i]))
                self._substeps[t_n_i][t_m_i + 1] = _trans[0] * _nodes[t_m_i] + _trans[1]
            # END FOR t_m_i

            self._substeps[t_n_i][-1] = _t_n + self._dt_n

#             print("__sol[0][" + str(t_n_i) + "] = " + str(self.solution[0][t_n_i]))
        # END FOR t_n_i

        if self.verbosity > 1:
            print("_substeps:\n" + str(self._substeps))
            if self.verbosity > 3:
                print("Initial Solution:\n" + str(self.__sol[0]))


        # #
        # # Compute SDC iterations
        # #
        # sdc iterations
        for k in range(1, self.iterations + 1):
            if self.verbosity > 0:
                print(80 * '*' + "\nIteration {:d}/{:d}:".format(k, self.iterations))
            # start with the initial value
            self.__sol[k][0][0] = self.__sol[k - 1][0][0]

            # iterate over steps
            for t_n_i in range(0, self.time_steps):
                # compute time point for this step
                _t_n = self.time_range[0] + t_n_i * self._dt_n
                if self.verbosity > 1:
                    print("  " + 10 * '-' + "\n  Time Step {:d} (_t_n={: f}):".format(t_n_i, _t_n))

                # copy last value from previous substep
                if t_n_i > 0:
                    self.__sol[k][t_n_i][0] = self.__sol[k][t_n_i - 1][-1]

                # iterate over substeps
                for t_m_i in range(1, len(self._substeps[t_n_i])):
                    # get time point for this substep
                    _t_m = self._substeps[t_n_i][t_m_i]
                    # compute delta t for this substep
                    _dt_m = _t_m - self._substeps[t_n_i][t_m_i - 1]
                    if self.verbosity > 2:
                        print("      Substep {:d} (_t_m={: f}, _dt_m={: f}):".format(t_m_i, _t_m, _dt_m))

                    # dummy assertion
                    assert _dt_m > 0.0, "dt_m should be larger 0 ({:d})".format(_dt_m)

                    # compute Eqn. 2.7 in explicit form
                    self.__sol[k][t_n_i][t_m_i] = self.__sol[k][t_n_i][t_m_i - 1] + \
                        _dt_m * (self.fnc(_t_m, t_m_i - 1) - self.fnc(_t_m, t_m_i)) + \
                        Gauss.integrate(func=self.fnc, t=_t_m, begin=_t_n, end=(_t_n + self._dt_n), nPoints=self.num_substeps, partial=t_m_i)
                    if self.verbosity > 3:
                        print("          sol = {: f} = {: f} + {: f} * ( {: f} - {: f} ) + {: f}".format(self.__sol[k][t_n_i][t_m_i], self.__sol[k][t_n_i][t_m_i - 1], _dt_m, self.fnc(_t_m, t_m_i - 1), self.fnc(_t_m, t_m_i), Gauss.integrate(func=self.fnc, t=_t_m, begin=_t_n, end=(_t_n + self._dt_n), nPoints=self.num_substeps, partial=t_m_i)))

                    # calculate error and error reduction
                    self._error[k][t_n_i][t_m_i] = abs(self.__sol[k][t_n_i][t_m_i] - self.exact(_t_m))
                    self._reduction[k - 1][t_n_i][t_m_i] = abs(self._error[k][t_n_i][t_m_i] / self._error[k - 1][t_n_i][t_m_i])
                # END FOR t_m_i

                if self.verbosity > 1:
                    print("  Solution:\n    t_m_i\t     t    \t    x(t) \treduction\t   exact \t   error")
                    for t_m_i in range(0, len(self._substeps[t_n_i])):
                       _t_m = self._substeps[t_n_i][t_m_i]
                       print('      {:d}    \t{: f}\t{: f}\t{: f}\t{: f}\t{: f}'.format(t_m_i, _t_m, self.__sol[k][t_n_i][t_m_i], self._reduction[k - 1][t_n_i][t_m_i], self.exact(_t_m), self._error[k][t_n_i][t_m_i]))
            # END FOR t_n_i

            if self.verbosity > 0:
                print("Overall reduction for this iteration: {:f}".format(self._reduction[k - 1].mean()))
        # END FOR k

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

    def print_solution(self):
        """
        """
        print("Solution after " + str(self.iterations) + " iterations:\n(t_n_i, t_m_i)\t     t    \t    x(t) \treduction\t   exact \t   error")
        for t_n_i in range(0, self.time_steps):
            _t_n = self.time_range[0] + t_n_i * self._dt_n
            for t_m_i in range(0, len(self._substeps[t_n_i])):
               _t_m = self._substeps[t_n_i][t_m_i]
               error = abs(self.__sol[-1][t_n_i][t_m_i] - self.exact(_t_m))
               print('    ({:d}, {:d})    \t{: f}\t{: f}\t{: f}\t{: f}\t{: f}'.format(t_n_i, t_m_i, _t_m, self.__sol[-1][t_n_i][t_m_i], self._reduction[-1][t_n_i][t_m_i], self.exact(_t_m), error))

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
