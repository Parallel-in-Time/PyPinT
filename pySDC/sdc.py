"""
SDC Algorithm
"""

import numpy as np
import pySDC.globals as Config
from pySDC.integrate.gauss import Gauss


class SDC(object):
    """
    General Provider for the SDC algorithm
    """

    def __init__(self):
        """
        Initialization
        """
        self.__function = lambda t, phi_t: -1.0
        self.__exact = lambda t: -t + 1.0
        self.__initialValue = 1.0
        self.__timeRange = [0.0, 1.0]
        self.__timeSteps = 1
        self.__numSubsteps = 3
        self.__iterations = 2
        self.__sol = np.zeros((1, 1, 1), dtype=float)
        self._substeps = np.zeros((self.time_steps, self.num_substeps + 2),
                                  dtype=float)
        self._dt_n = float(self.time_range[1] - self.time_range[0]) / \
            float(self.time_steps)
        self._relred = np.zeros((1, 1, 1), dtype=float)
        self._error = np.zeros((1, 1, 1), dtype=float)
        self.verbosity = 4

    def solve(self, integrator="lobatto", initial="copy"):
        """
        solves a given problem setup

        :param integrator: integration method used as integrator on substeps
        :param initial:
        :raises:
        """
        # determine number of integration points per time step in dependency of
        #  integration method and whether it uses the interval borders as
        #  integration points (as Gauss-Lobatto) or not (as Gauss-Legendre)
        if integrator == "lobatto":
            # Gauss-Lobatto uses outer interval points as integration points
            n_nodes = self.num_substeps + 1
            n_sub_values = n_nodes
        elif integrator == "legendre":
            # Gauss-Legendre does not use outer interval points as integration
            #  points
            # TODO: we need to interpolate interval borders with Gauss-Legendre
            n_nodes = self.num_substeps - 1
            n_sub_values = n_nodes + 2
            raise NotImplementedError("Gauss-Legendre integration not yet " +
                                      "implemented.")
        else:
            raise ValueError("No known integrator given: {s}"
                             .format(integrator))

        self.init_solution(integrator, initial, n_nodes, n_sub_values)

        Config.LOG.debug("Sub Steps: {0}".format(str(self._substeps)))
        Config.LOG.debug("Initial Solution:\n{0}".format(str(self.__sol[0])))

        #####
        ## Compute SDC iterations
        # SDC iterations
        for k in range(1, self.iterations + 1):
            Config.LOG.info("")
            Config.LOG.info("Iteration {:d}/{:d}:".format(k, self.iterations))

            # copy initial value from previous SDC iteration
            self.__sol[k][0][0] = self.__sol[k - 1][0][0]

            # iterate over coarse time steps
            for t_n_i in range(0, self.time_steps):
                self.sdc_sweep(k, t_n_i, n_nodes, n_sub_values, integrator)
            # END FOR t_n_i
        # END FOR k

    def init_solution(self, integrator, initial, n_nodes, n_sub_values):
        """

        :param integrator:
        :param initial:
        :param n_nodes:
        :param n_sub_values:
        :return:
        """
        # initialize solution array
        self.__sol = np.zeros((self.iterations + 1, self.time_steps,
                               n_sub_values), dtype=float)
        self._error = np.zeros((self.iterations + 1, self.time_steps,
                                n_sub_values), dtype=float)
        # multi-dimensional array for relative reduction; all 1-based
        #  indizes:
        #    1.: iteration
        #    2.: coarse time step
        #    3.: sub step in coarse time step
        self._relred = np.zeros((self.iterations + 1, self.time_steps,
                                 n_sub_values), dtype=float)

        # matrix for time points of all integration points
        self._substeps = np.zeros((self.time_steps, n_sub_values), dtype=float)

        # delta of coarse time steps
        self._dt_n = float(self.time_range[1] - self.time_range[0]) / \
            float(self.time_steps)

        # calculate integration nodes for optimal substep alignment
        Config.LOG.debug("[{: f}, {: f}], {: f} ==> {: f}"
                         .format(self.time_range[0], self.time_range[1],
                                 self.time_steps, self._dt_n))
        _nodes = Gauss.get_nodes_and_weights(n_nodes, integrator)['nodes']

        #####
        ## Set initial values and compute substep points
        Config.LOG.debug("Preparing initial values.")

        self._substeps[0][0] = self.time_range[0]
        self._substeps[-1][-1] = self.time_range[1]

        # iterate over coarse time steps
        for t_n_i in range(0, self.time_steps):
            # calculate current time point
            _t_n = self.time_range[0] + t_n_i * self._dt_n

            # transform [t_n, t_n+_dt_n] into Gauss nodes
            _trans = Gauss.transform(_t_n, _t_n + self._dt_n)
            assert len(_trans) == 2, \
                "Coordinate transformation failed (len(_trans)={:d})." \
                .format(len(_trans))
            assert len(self._substeps) > t_n_i, \
                "Substeps not correctly initialized (len(_substeps)={:d})." \
                .format(len(self._substeps))

            # initialize solution vector for this coarse time step
            self.__sol[0][t_n_i] = np.asarray([self.initial_value] *
                                              n_sub_values)

            if t_n_i > 0:
                if initial == "euler":
                    # compute initial values for _t_n via Standard Euler
                    self.__sol[0][t_n_i][0] = self.__sol[0][t_n_i - 1][0] + \
                        self._dt_n * self.fnc(_t_n)
                elif initial == "copy":
                    # set global initial value as initial value for current
                    #  coarse time step
                    self.__sol[0][t_n_i][0] = self.initial_value
                else:
                    raise ValueError("Given method for broadcasting " +
                                     "initial values not known: {s}"
                                     .format(initial))

                # make sure the start of this coarse time step equals the end
                #  point of the previous coarse time step
                assert self._substeps[t_n_i][0] == \
                    self._substeps[t_n_i - 1][-1], \
                    "Start of this coarse time step not end point of " + \
                    "previous coarse time step: {: f} != {: f}" \
                        .format(self._substeps[t_n_i][0],
                                self._substeps[t_n_i - 1][-1])

            # compute substep points for current coarse time step
            for t_m_i in range(0, n_sub_values):
                #print("  t_m_i={:d}".format(t_m_i))
                assert len(_nodes) > t_m_i, "Fever nodes than steps"
                if integrator == "lobatto":
                    # Gauss-Lobatto uses integration borders as integration
                    #  nodes make sure they are correct ...
                    if t_m_i == 0:
                        # (beginning of substep)
                        assert self._substeps[t_n_i][0] == \
                            _trans[0] * _nodes[0] + _trans[1], \
                            "First substep time point not equal first " + \
                            "integration node: {: f} != {: f}" \
                            .format(self._substeps[t_n_i][0],
                                    _trans[0] * _nodes[0] + _trans[1])
                    elif t_m_i == n_sub_values - 1:
                        # (end of substep)
                        assert self._substeps[t_n_i][-1] == \
                            _trans[0] * _nodes[-1] + _trans[1], \
                            "Last substep time point not equal last " + \
                            "integration node: {: f} != {: f}" \
                            .format(self._substeps[t_n_i][-1],
                                    _trans[0] * _nodes[-1] + _trans[1])
                    else:
                        # ... and calculate intermediate nodes only
                        self._substeps[t_n_i][t_m_i] = _trans[0] * \
                            _nodes[t_m_i] + _trans[1]
                elif integrator == "legendre":
                    # Gauss-Legendre only uses inner interval points as
                    #  integration nodes
                    self._substeps[t_n_i][t_m_i + 1] = _trans[0] * \
                        _nodes[t_m_i] + _trans[1]
                else:
                    # will not reach this, as it has raised previously
                    pass
            # END FOR t_m_i

            # calculate end point of this coarse time step
            self._substeps[t_n_i][-1] = _t_n + self._dt_n

        # END FOR t_n_i

    def sdc_sweep(self, k, t_n_i, n_nodes, n_sub_values, integrator):
        """

        :param k:
        :param t_n_i:
        :param n_nodes:
        :param n_sub_values:
        :param integrator:
        :return:
        """
        # compute starting coarse time point for this step
        _t_n = self.time_range[0] + t_n_i * self._dt_n
        Config.LOG.info("  Time Step {:d} (_t_n={: f}):"
                        .format(t_n_i, _t_n))

        # in case it is not the first coarse time step copy last value
        #  from previous coarse time step as initial value for this
        #  coarse time step
        if t_n_i > 0:
            self.__sol[k][t_n_i][0] = self.__sol[k][t_n_i - 1][-1]

        # iterate over substeps
        #  (of cause: skip initial value)
        for t_m_i in range(1, self.num_substeps + 1):
            self.sdc_step(k, t_n_i, _t_n, t_m_i, n_nodes, n_sub_values, integrator)
        # END FOR t_m_i

        Config.LOG.info("Solution after iteration {:d}:\n".format(k) +
                        "t_m_i\t     t    \t    x(t) \treduction   |" +
                        "\t   exact \t   error")

        # compute error and relative reduction
        #  (thus iterate over substeps again)
        for t_m_i in range(0, n_sub_values):
            # query time point for this substep
            _t_m = self._substeps[t_n_i][t_m_i]

            # calculate absolute error
            self._error[k][t_n_i][t_m_i] = abs(
                self.__sol[k][t_n_i][t_m_i] - self.exact(_t_m))
            # ... and relative error reduction
            if t_m_i > 0:
                self._relred[k][t_n_i][t_m_i] = \
                    SDC.calc_rel_err_reduction(
                        self._error[k - 1][t_n_i][t_m_i],
                        self._error[k][t_n_i][t_m_i])
            if self.verbosity > 1:
                if k > 1:
                    print("      {:d}    \t{: f}\t{: f}\t{: f}   |"
                          .format(t_m_i, _t_m,
                                  self.__sol[k][t_n_i][t_m_i],
                                  self._relred[k][t_n_i][t_m_i]) +
                          "\t{: f}\t{: f}"
                          .format(self.exact(_t_m),
                                  self._error[k][t_n_i][t_m_i]))
                else:
                    print("      {:d}    \t{: f}\t{: f}\t"
                          .format(t_m_i, _t_m,
                                  self.__sol[k][t_n_i][t_m_i]) +
                          "            |\t{: f}\t{: f}"
                          .format(self.exact(_t_m),
                                  self._error[k][t_n_i][t_m_i]))
        # END FOR t_m_i

    def sdc_step(self, k, t_n_i, t_n, t_m_i, n_nodes, n_sub_values, integrator):
        """

        :param k:
        :param t_n_i:
        :param t_n:
        :param t_m_i:
        :param n_nodes:
        :param n_sub_values:
        :param integrator:
        :return:
        """
        # query time point for this substep
        _t_m = self._substeps[t_n_i][t_m_i]

        # ... and previous substep
        _t_m_p = self._substeps[t_n_i][t_m_i - 1]

        # compute delta t for this substep
        _dt_m = _t_m - _t_m_p
        Config.LOG.info("    Substep {:d} (_t_m={: f}, "
                        .format(t_m_i, _t_m) +
                        "_dt_m={: f}):".format(_dt_m))

        # make sure nothing goes really wrong
        assert _dt_m > 0.0, \
            "Delta of substep must be larger 0: {:f}".format(_dt_m)

        # gather values for integration
        _copy_mask = np.concatenate((np.asarray([True] * t_m_i),
                                     np.asarray([False] *
                                                (n_sub_values -
                                                 t_m_i))))
        Config.LOG.debug("_copy_mask ({:d} : {:d}) = {}"
                         .format(t_m_i, (n_sub_values - t_m_i),
                                 str(_copy_mask)))
        _integrate_values = np.where(_copy_mask,
                                     self.__sol[k][t_n_i],
                                     self.__sol[k - 1][t_n_i])
        Config.LOG.debug("_integrate_values = {}".format(
            str(_integrate_values)))

        # integrate this substep
        integral = Gauss.integrate(func=None,
                                   vals=_integrate_values,
                                   begin=t_n,
                                   end=(t_n + self._dt_n),
                                   n=n_nodes, partial=t_m_i,
                                   method=integrator)
        # compute new solution for this substep
        #  (cf. Minion, Eqn. 2.7, explicit form)
        self.__sol[k][t_n_i][t_m_i] = \
            self.__sol[k][t_n_i][t_m_i - 1] + _dt_m * (
                self.fnc(_t_m_p, self.__sol[k][t_n_i][t_m_i - 1]) -
                self.fnc(_t_m_p,
                         self.__sol[k - 1][t_n_i][t_m_i])) \
            + _dt_m * integral
        Config.LOG.debug("{}sol = {: f} = {: f} + {: f} "
                         .format(' ' * 10,
                                 self.__sol[k][t_n_i][t_m_i],
                                 self.__sol[k][t_n_i][t_m_i - 1],
                                 _dt_m) +
                         "* ( {: f} - {: f} ) + {: f} * {: f}"
                         .format(self.fnc(_t_m_p,
                                          self.__sol[k][t_n_i][t_m_i - 1]),
                                 self.fnc(_t_m_p,
                                          self.__sol[k - 1][t_n_i][t_m_i]),
                                 _dt_m, integral))

    @staticmethod
    def calc_rel_err_reduction(error1, error2):
        """
        :param error1:
        :param error2:
        :return:
        """
        if error1 == error2:
            return 1.0
        elif error1 == 0.0 or error2 == 0.0:
            return 0.0
        else:
            return float(abs(float(error2) / float(error1)))

    @property
    def solution(self):
        """
        solution of the last call to :py:func:`SDC.solve`

        :rtype: multi-dimensional List of decimal.Decimal
        """
        return self.__sol

    @solution.deleter
    def solution(self):
        """
        resets SDC.solution
        """
        del self.__sol

    def print_solution(self):
        """
        prints current solution
        """
        print("Solution after {:d} iterations:".format(self.iterations))
        print("(t_n_i, t_m_i)\t     t    \t    x(t) \tover.red.   |" +
              "\t   exact \t   error")
        for t_n_i in range(0, self.time_steps):
            _t_n = self.time_range[0] + t_n_i * self._dt_n
            for t_m_i in range(0, len(self._substeps[t_n_i])):
                _t_m = self._substeps[t_n_i][t_m_i]
                error = abs(self.__sol[-1][t_n_i][t_m_i] - self.exact(_t_m))
                print("    ({:d}, {:d})    \t{: f}\t{: f}\t{: f}   |"
                      .format(t_n_i, t_m_i, _t_m, self.__sol[-1][t_n_i][t_m_i],
                              SDC.calc_rel_err_reduction(
                                  self._error[1][t_n_i][t_m_i],
                                  self._error[-1][t_n_i][t_m_i])) +
                      "\t{: f}\t{: f}"
                      .format(self.exact(_t_m), error))

    @property
    def fnc(self):
        """
        function describing the problem

        :rtype: function pointer
        """
        return self.__function

    @fnc.setter
    def fnc(self, function):
        """
        sets function
        :param function:
        :return:
        """
        self.__function = function

    @fnc.deleter
    def fnc(self):
        """
        resets SDC.fnc
        :return:
        """
        del self.__function

    @property
    def exact(self):
        """
        exact solution function of the problem

        Returns
        -------
        function pointer
        """
        return self.__exact

    @exact.setter
    def exact(self, function):
        """
        sets exact solution function
        :param function:
        :return:
        """
        self.__exact = function

    @exact.deleter
    def exact(self):
        """
        resets SDC.exact

        :return:
        """
        del self.__exact

    @property
    def initial_value(self):
        """
        initial value of the problem

        :rtype: decimal.Decimal
        """
        return self.__initialValue

    @initial_value.setter
    def initial_value(self, value):
        """
        sets initial value
        :param value:
        :return:
        """
        self.__initialValue = value

    @initial_value.deleter
    def initial_value(self):
        """
        resets SDC.initial_value
        :return:
        """
        del self.__initialValue

    @property
    def time_range(self):
        """
        pair of start and end time

        :rtype: List of two decimal.Decimal

        :raises: ValueError (on setting if time range is non-positive or zero)
        """
        return self.__timeRange

    @time_range.setter
    def time_range(self, value):
        """
        sets time_range
        :param value:
        :return:
        """
        if value[1] <= value[0]:
            raise ValueError("Time interval must be non-zero positive " +
                             "[start, end]: [{:f }, {: f}]"
                             .format(value[0], value[1]))
        self.__timeRange = [value[0], value[1]]

    @time_range.deleter
    def time_range(self):
        """
        resets SDC.time_range
        :return:
        """
        del self.__timeRange

    @property
    def time_steps(self):
        """
        number of time steps

        :rtype: Integer

        :raises: ValueError (on setting if number steps is not positive)
        """
        return self.__timeSteps

    @time_steps.setter
    def time_steps(self, value):
        """
        sets number of time steps

        :param value:
        :return:
        """
        if value <= 0:
            raise ValueError("At least one time step is neccessary.")
        self.__timeSteps = value

    @time_steps.deleter
    def time_steps(self):
        """
        resets SDC.time_steps
        :return:
        """
        del self.__timeSteps

    @property
    def num_substeps(self):
        """
        number of substeps of each time step

        :rtype: Integer

        :raises: ValueError (on setting if number substeps is not positive)
        """
        return self.__numSubsteps

    @num_substeps.setter
    def num_substeps(self, value):
        """
        sets number of substeps per time step
        :param value:
        :return:
        """
        if value <= 0:
            raise ValueError("At least one substep is neccessary: {:d}"
                             .format(value))
        self.__numSubsteps = value

    @num_substeps.deleter
    def num_substeps(self):
        """
        resets SDC.num_substeps
        :return:
        """
        del self.__numSubsteps

    @property
    def iterations(self):
        """
        number if SDC iterations

        :rtype: Integer

        :raises: ValueError (on setting if iterations is not positive)
        """
        return self.__iterations

    @iterations.setter
    def iterations(self, value):
        """
        sets number of iterations
        :param value:
        :return:
        """
        if value <= 0:
            raise ValueError("At least one iteration is neccessary.")
        self.__iterations = value

    @iterations.deleter
    def iterations(self):
        """
        resets SDC.iterations
        :return:
        """
        del self.__iterations
