# coding=utf-8
"""
.. moduleauthor:: Torbjörn Klatt <t.kaltt@fz-juelich.de>
.. moduleauthor:: Dieter Moser <d.moser@fz-juelich.de>
"""
import numpy as np

from pypint.plugins.multigrid.i_transient_multigrid_problem import ITransientMultigridProblem
from pypint.plugins.multigrid.i_multigrid_level import IMultigridLevel
from pypint.plugins.multigrid.stencil import Stencil
from pypint.utilities import assert_named_argument, assert_is_key, assert_is_instance
from pypint.utilities.logging import LOG, this_got_called


class HeatEquation(ITransientMultigridProblem):
    """A parabolic partial differential equation in two spacial dimensions

    Also known as the 2D-Heat-Equation:

    .. math::

        \\frac{\\partial u(x,y,t)}{\\partial t} = \\alpha \\laplace u(x,y,t)

    with the thermal diffusivity :math:`\\alpha`.
    """
    def __init__(self, *args, **kwargs):
        super(HeatEquation, self).__init__(*args, **kwargs)

        # HasExactSolutionMixin.__init__(self, *args, **kwargs)

        self._thermal_diffusivity = kwargs.get('thermal_diffusivity', -1.0)

        if self.time_start is None:
            self.time_start = 0.0
        if self.time_end is None:
            self.time_end = 1.0
        if self.initial_value is None:
            self.initial_value = np.zeros(self.dim_for_time_solver)

        if isinstance(self.thermal_diffusivity, complex):
            self.numeric_type = np.complex

        self._mg_stencil = kwargs.get('mg_stencil')
        self._mg_level = kwargs.get('mg_level')
        self._direct_solvers = {}

        if kwargs.get('delta_times_for_time_levels') is not None and self._mg_level is not None:
            assert_is_instance(kwargs['delta_times_for_time_levels'], (list, np.ndarray),
                               descriptor="Delta Times for Time Levels", checking_obj=self)
            for time_level in kwargs['delta_times_for_time_levels']:
                assert_is_instance(kwargs['delta_times_for_time_levels'][time_level], (list, np.ndarray),
                                   descriptor="Delta Times for Time Level %d" % time_level, checking_obj=self)
                for delta_time in kwargs['delta_times_for_time_levels'][time_level]:
                    self.initialize_direct_space_solver(time_level, delta_time, kwargs['mg_level'])

    @property
    def thermal_diffusivity(self):
        return self._thermal_diffusivity

    @thermal_diffusivity.setter
    def thermal_diffusivity(self, value):
        self._thermal_diffusivity = value

    def evaluate_wrt_time(self, time, phi_of_time, **kwargs):
        this_got_called(self, time=time, phi_of_time=phi_of_time, **kwargs)
        if kwargs.get('partial'):
            if kwargs['partial'] == "impl":
                self._mg_level.mid[:] = phi_of_time.reshape(-1)
                _padded_phi = self._mg_level.evaluable_view(self._mg_stencil)
                LOG.debug("padded phi: %s" % _padded_phi)
                _out = self._mg_stencil.eval_convolve(_padded_phi)
                LOG.debug("  ==> %s" % _out.reshape(phi_of_time.shape))
                return _out.reshape(phi_of_time.shape)
            else:
                return np.zeros(phi_of_time.shape)
        else:
            self._mg_level.mid[:] = phi_of_time.reshape(-1)
            _padded_phi = self._mg_level.evaluable_view(self._mg_stencil)
            LOG.debug("padded phi: %s" % _padded_phi)
            _out = self._mg_stencil.eval_convolve(_padded_phi)
            LOG.debug("  ==> %s" % _out.reshape(phi_of_time.shape))
            return _out.reshape(phi_of_time.shape)

    def mg_stencil(self, delta_time, delta_space):
        _stencil = (-1.0 * delta_time * self.thermal_diffusivity) * np.array([1.0, -2.0 - (delta_space**2 / delta_time), 1.0]) \
            / (delta_space**2)
        LOG.debug("Stencil for dt=%f, h=%f: %s" % (delta_time, delta_space, _stencil))
        return _stencil

    def initialize_direct_space_solver(self, time_level, delta_time, mg_level=None):
        if mg_level is None:
            mg_level = self._mg_level
        assert_is_instance(mg_level, IMultigridLevel, descriptor="Multigrid Level", checking_obj=self)
        _stencil = Stencil(self.mg_stencil(delta_time, mg_level.h))
        time_level = str(time_level)
        delta_time = str(delta_time)
        if self._direct_solvers.get(time_level) is None:
            LOG.debug("Initializing Solvers for Time Level '%s'" % time_level)
            self._direct_solvers[time_level] = {}
        LOG.debug("  Initializing Solver for Time Level '%s' and Delta Node '%s'" % (time_level, delta_time))
        LOG.debug("    shape: %s" % (mg_level.mid.shape))
        self._direct_solvers[time_level][delta_time] = {
            'mg_level': mg_level,
            'stencil': _stencil,
            'solver': _stencil.generate_direct_solver(mg_level.mid.shape)
        }

    def implicit_solve(self, next_x, func, method="direct", **kwargs):
        """Space-Solver for the Heat Equation

        Parameters
        ----------
        next_x : :py:class:`numpy.ndarray`
            initial value for the MG solver
        func :
            *unused*
        method : :py:class:`str`
            method specifying the space solver;
            one of ``mg`` or ``direct`` (default)
        time_level : :py:class:`int`
            time level in MLSDC-notation (i.e. 0 is base level of MLSDC)
        delta_time : :py:class:`float`
            distance from the previous to currently calculated time node
        """
        this_got_called(self, next_x=next_x, func=func, **kwargs)
        assert_named_argument('expl_term', kwargs, types=np.ndarray, descriptor="RHS for Space Solver",
                              checking_obj=self)
        # assert_named_argument('time_level', kwargs, types=int, descriptor="Time Level", checking_obj=self)
        if kwargs.get('time_level') is None:
            kwargs['time_level'] = 0
        time_level = str(kwargs['time_level'])
        assert_named_argument('delta_time', kwargs, types=float, descriptor="Delta Time Node", checking_obj=self)
        delta_time = str(kwargs['delta_time'])
        # assert_is_key(time_level, self._direct_solvers, key_desc="Time Level", dict_desc="Direct Solvers",
        #               checking_obj=self)
        # assert_is_key(delta_time, self._direct_solvers[time_level],
        #               key_desc="Delta Time '%s'" % delta_time,
        #               dict_desc="Direct Solvers for Time Level '%s'" % 'time_level', checking_obj=self)
        if time_level not in self._direct_solvers or delta_time not in self._direct_solvers[time_level]:
            self.initialize_direct_space_solver(kwargs['time_level'], kwargs['delta_time'])
        _this_set = self._direct_solvers[time_level][delta_time]
        _this_set['mg_level'].rhs = kwargs['expl_term']

        _this_set['stencil'].modify_rhs(_this_set['mg_level'])
        LOG.debug("modified RHS: %s" % _this_set['mg_level'].rhs)
        _sol = self.mg_solve(_this_set['mg_level'].rhs, method='direct', solver=_this_set['solver'])
        LOG.debug("Implicit Solve => %s" % _sol)
        return _sol

    def print_lines_for_log(self):
        _lines = super(HeatEquation, self).print_lines_for_log()
        _lines['Formula'] = r"d u(x,t) / dt = \alpha \laplace u(x,t)"
        _lines['Coefficients'] = {
            r'\alpha': self.thermal_diffusivity
        }
        return _lines

    def __str__(self):
        return r"\partial u(x,t) / \partial t = \alpha \laplace u(x,t), \alpha=%s" % self.thermal_diffusivity


__all__ = ['HeatEquation']
