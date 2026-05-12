"""Public API for the CO2 diffusivity package."""

from .readfile import load_literature, load_loading_series, load_msd_file
from .msd import calculate_diffusivity, detect_diffusive_regime
from .calculate import calculate_single
from .regression import fit_anisotropic, svd_linear_fit, svd_linear_fit_with_residuals
from .models import compare_models, fit_exponential, fit_linear, fit_powerlaw
from .uncertainty import (
    block_average_diffusivity,
    combined_uncertainty,
)
from .export import save_all, save_csv, save_json
from . import plot

__version__ = '0.0.1'

__all__ = [
    'calculate_single',
    'load_msd_file',
    'load_loading_series',
    'load_literature',
    'calculate_diffusivity',
    'detect_diffusive_regime',
    'svd_linear_fit',
    'svd_linear_fit_with_residuals',
    'fit_anisotropic',
    'fit_powerlaw',
    'fit_exponential',
    'fit_linear',
    'compare_models',
    'block_average_diffusivity',
    'combined_uncertainty',
    'save_json',
    'save_csv',
    'save_all',
    'plot',
]
