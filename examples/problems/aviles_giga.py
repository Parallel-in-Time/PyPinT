# coding=utf-8
import numpy as np
import scipy.fftpack as spfft
from pypint.plugins.multigrid.i_transient_multigrid_problem import IInitialValueProblem
from pypint.problems.has_direct_implicit_mixin import HasDirectImplicitMixin
from pypint.utilities import assert_named_argument, assert_is_instance


class AvilesGiga(IInitialValueProblem):
    """A nonlinear partial differential equation in two spacial dimensions for
        which describes a homogeneous dipole model


    .. math::

        E(u) = \\int_{\\Omega} \\frac{\\epsilon}{2} (\\grad^2 u)^2 + \frac{1}{4\\epsilon}(1 - |\\grad i|^2)^2 dx

    smoothness factor:math:`\\epsilon`.
    """
    def __init__(self, *args, **kwargs):
        self._n = kwargs.get('n')
        self._m = 2*self._n + 1
        kwargs.update({"dim": (self._m, self._m, 1)})

        super(AvilesGiga, self).__init__(*args, **kwargs)

        # HasDirectImplicitMixin.__init__(self, *args, **kwargs)

        self._epsilon = kwargs.get('epsilon', 1.0)

        if self.time_start is None:
            self.time_start = 0.0
        if self.time_end is None:
            self.time_end = 1.0
        if self.initial_value is None:
            self.initial_value = np.random.rand(self.dim_for_time_solver)

        if isinstance(self.epsilon, complex):
            self.numeric_type = np.complex


        self._u = np.zeros((self._m, self._m), dtype=np.complex128)

        # place to work on
        self._u_x = self._u.copy()
        self._u_y = self._u.copy()
        self._u_f = self._u.copy()
        self._u_angle = self._u.copy()
        self._u_edens = self._u.copy()
        self._B = self._u.copy()
        # arrays to work in fourier space
        self._k_od = np.hstack((np.arange(self._n+1, dtype=np.int64),
                               np.arange(self._n, dtype=np.int64)-self._n))
        self._k_y = self._k_od
        self._k_x = self._k_od.reshape(self._m, 1)
        for i in range(self._m-1):
            self._k_y = np.vstack((self._k_y, self._k_od))
            self._k_x = np.hstack((self._k_x, self._k_od.reshape(self._m, 1)))

        self._k_2 = self._k_x**2 + self._k_y**2
        self._k_4 = self._k_2**2


        if kwargs.get('delta_times_for_time_levels') is not None and self._mg_level is not None:
            assert_is_instance(kwargs['delta_times_for_time_levels'], (list, np.ndarray),
                               descriptor="Delta Times for Time Levels", checking_obj=self)
            for time_level in kwargs['delta_times_for_time_levels']:
                assert_is_instance(kwargs['delta_times_for_time_levels'][time_level], (list, np.ndarray),
                                   descriptor="Delta Times for Time Level %d" % time_level, checking_obj=self)
                for delta_time in kwargs['delta_times_for_time_levels'][time_level]:
                    self.initialize_direct_space_solver(time_level, delta_time, kwargs['mg_level'])

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        self._epsilon = value

    def fft(self, u):
        """Wrapper around some fft
        """
        return spfft.fft2(u)

    def ifft(self, u):
        """Wrapper around some ifft
        """
        return spfft.ifft2(u)

    def energy_linear(self):
        pass

    def energy_non_linear(selfself):
        pass

    def compute_grad(self):
        self._u_x[:] = np.real(self.ifft(np.complex(0, 1) * self._k_x * self._u_f))
        self._u_y[:] = np.real(self.ifft(np.complex(0, 1) * self._k_y * self._u_f))
        self._B = self._u_x**2 + self._u_y**2 - 1

    def compute_non_linear(self):
        return np.real(self.ifft(np.complex(0, 1) * self._k_x * (self.fft(self._B*self._u_x)) +
                                 np.complex(0, 1) * self._k_y * (self.fft(self._B*self._u_y))))

    def compute_linear(self):
        return np.real(self.ifft(self._k_4 * self.fft(self._u)))

    def evaluate_wrt_time(self, time, phi_of_time, **kwargs):
        """RHS ...
        """
        super(AvilesGiga, self).evaluate_wrt_time(time, phi_of_time, **kwargs)
        self._u.reshape(-1)[:] = phi_of_time
        self._u_f = self.fft(self._u)
        self.compute_grad()
        if kwargs.get('partial') is not None:
            if isinstance(kwargs['partial'], str) and kwargs['partial'] == 'impl':
                return (- self.epsilon * self.compute_linear()).flatten()
            elif kwargs['partial'] == 'expl':
                return (self.compute_non_linear() / self.epsilon).flatten()
        else:
            return (self.compute_non_linear() / self.epsilon - self.epsilon * self.compute_linear()).flatten()

    def print_lines_for_log(self):
        _lines = super(AvilesGiga, self).print_lines_for_log()
        _lines['Formula'] = r"d u(x,t) / dt = -\epsilon \laplace^2 u(x,t) + (\grad \cdot ((|grad u|^2-1)\grad u))"
        _lines['Coefficients'] = {
            r'\epsilon': self.epsilon
        }
        return _lines

    def __str__(self):
        return r"\partial u(x,t) / \partial t = \alpha \laplace u(x,t), \alpha=%s" % self.thermal_diffusivity


__all__ = ['AvilesGiga']
