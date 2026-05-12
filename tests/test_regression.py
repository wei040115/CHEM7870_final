import numpy as np
import pytest
from diffusivity.regression import svd_linear_fit, svd_linear_fit_with_residuals, fit_anisotropic


def test_svd_recovers_slope_and_intercept():
    """Exact slope and intercept recovered on noise-free data."""
    t = np.linspace(0, 1e6, 500)
    y = 0.15 * t + 5.0
    slope, intercept, r2 = svd_linear_fit(t, y)
    assert abs(slope - 0.15) / 0.15 < 1e-8
    assert abs(intercept - 5.0) / 5.0 < 1e-8
    assert abs(r2 - 1.0) < 1e-10


def test_svd_noisy_r2_and_slope():
    """Noisy data: R² > 0.99 and slope within 2% of true value."""
    rng = np.random.default_rng(0)
    t = np.linspace(0, 2e6, 2000)
    y = 0.025 * t + rng.normal(0, 50, len(t))
    slope, _, r2 = svd_linear_fit(t, y)
    assert r2 > 0.99
    assert abs(slope - 0.025) / 0.025 < 0.02


def test_svd_with_residuals_shapes_and_consistency():
    """Residuals and y_pred shapes are correct; y_pred matches simple fit."""
    rng = np.random.default_rng(42)
    t = np.linspace(0, 1e6, 300)
    y = 0.1 * t + rng.normal(0, 20, 300)
    slope1, intercept1, _ = svd_linear_fit(t, y)
    _, _, _, residuals, y_pred = svd_linear_fit_with_residuals(t, y)
    assert residuals.shape == (300,)
    np.testing.assert_allclose(slope1 * t + intercept1, y_pred, rtol=1e-10)


def test_fit_anisotropic_recovers_known_D():
    """D_x, D_y, D_z recovered within 0.001% on noise-free data."""
    t = np.linspace(0, 2e6, 1000)
    Dx, Dy, Dz = 0.02, 0.03, 0.05
    msd_x = 2 * Dx * t
    msd_y = 2 * Dy * t
    msd_z = 2 * Dz * t
    res = fit_anisotropic(t, msd_x, msd_y, msd_z)
    assert abs(res['D_x'] - Dx) / Dx < 1e-6
    assert abs(res['D_y'] - Dy) / Dy < 1e-6
    assert abs(res['D_z'] - Dz) / Dz < 1e-6


def test_fit_anisotropic_output_keys():
    t = np.linspace(1, 1e6, 200)
    msd = 0.02 * t
    res = fit_anisotropic(t, msd, msd, msd)
    for key in ('slope_x', 'slope_y', 'slope_z',
                'intercept_x', 'intercept_y', 'intercept_z',
                'r2_x', 'r2_y', 'r2_z', 'D_x', 'D_y', 'D_z'):
        assert key in res
