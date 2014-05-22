# coding=utf-8

import numpy as np
from pypint.plugins.multigrid.interpolation import InterpolationByStencilForLevelsClassical
from pypint.plugins.multigrid.restriction import RestrictionByStencilForLevelsClassical
from pypint.plugins.multigrid.stencil import Stencil

MG_INTERPOLATION_PRESETS = {}
"""Useful presets for the interpolation operator used by MGCore
"""

MG_RESTRICTION_PRESETS = {}
"""Useful presets for the restriction operator used by MGCore
"""

MG_SMOOTHER_PRESETS = {}
"""Useful presets for the smoothers operators used by MGCore
"""

MG_LEVEL_PRESETS = {}
"""Useful presets for the level setup used by MGCore
"""

corner_array = np.ones((2., 2.)) * 0.25
border_arr_h = np.asarray([[0.5, 0.5]])
border_arr_v = np.asarray([[0.5], [0.5]])

ipl_stencil_list = [(Stencil(np.asarray([[1]])), (1, 1)),
                    (Stencil(corner_array), (0, 0)),
                    (Stencil(border_arr_h), (1, 0)),
                    (Stencil(border_arr_v), (0, 1))]

MG_INTERPOLATION_PRESETS["Standard-2D"] = {
    "ipl_class": InterpolationByStencilForLevelsClassical,
    "ipl_opts": [ipl_stencil_list]
}

MG_RESTRICTION_PRESETS["Standard-2D"] = {
    "rst_class": RestrictionByStencilForLevelsClassical,
    "rst_opts": [Stencil(np.asarray([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]])/16)]
}

center = np.asarray([0])

MG_INTERPOLATION_PRESETS["Standard-1D"] = {
    "ipl_class": InterpolationByStencilForLevelsClassical,
    "ipl_stencil_list": [(Stencil(np.asarray([1]), center), (1,)), (Stencil(np.asarray([0.5, 0.5]), center), (0,))],
    "ipl_opts": [[(Stencil(np.asarray([1]), center), (1,)), (Stencil(np.asarray([0.5, 0.5]), center), (0,))]]
}

MG_RESTRICTION_PRESETS["Standard-1D"] = {
    "rst_class": RestrictionByStencilForLevelsClassical,
    "rst_stencil": Stencil(np.asarray([0.25, 0.5, 0.25])),
    "rst_opts": [Stencil(np.asarray([0.25, 0.5, 0.25]))]
}

MG_LEVEL_PRESETS["Standard-1D"] = {
    "dim": 1,
    "shape_coarse": 32,
    "num_levels": 3,
    "max_borders": np.asarray([2, 2])
}

MG_LEVEL_PRESETS["Standard-2D"] = {
    "dim": 2,
    "shape_coarse": (16, 16),
    "num_levels": 3,
    "max_borders":  np.ones((2, 2)) * 2
}

MG_SMOOTHER_PRESETS["Jacobi"] = {
    "smoothing_type": "jacobi",
    "n_pre": 3,
    "n_post": 3,
    "smooth_opts": {
        "omega": 1.0
    }
}

MG_SMOOTHER_PRESETS["ILU"] = {
    "smoothing_type": "ilu",
    "n_pre": 3,
    "n_post": 3,
    "smooth_opts": {
        "fill_factor": 10,
        "drop_tolerance": 1e-6
    }
}

__all__ = ['MG_SMOOTHER_PRESETS', 'MG_LEVEL_PRESETS', 'MG_RESTRICTION_PRESETS', 'MG_INTERPOLATION_PRESETS']
