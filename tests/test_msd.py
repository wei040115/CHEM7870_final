import numpy as np
import pytest
from diffusivity.msd import calculate_diffusivity, detect_diffusive_regime

ANG2_FS_TO_M2_S = 1e-5


def _make_data(D=0.025, n=4001, noise=0.0, seed=0):
    """Synthetic MSD data: D_x = D_y = D_z = D so D_iso = D."""
    rng   = np.random.default_rng(seed)
    steps = np.arange(n) * 500
    t     = steps.astype(float)
    msd_x = 2 * D * t + rng.normal(0, noise, n)
    msd_y = 2 * D * t + rng.normal(0, noise, n)
    msd_z = 2 * D * t + rng.normal(0, noise, n)
    return np.column_stack([steps, msd_x, msd_y, msd_z, msd_x + msd_y + msd_z])


def test_recovers_D_total():
    """Isotropic D recovered within 2% (Einstein: D = slope_total / 6)."""
    D_true = 0.025
    data   = _make_data(D=D_true, noise=1.0)
    res    = calculate_diffusivity(data, timestep_fs=1.0)
    assert abs(res['D_m2s'] - D_true * ANG2_FS_TO_M2_S) / (D_true * ANG2_FS_TO_M2_S) < 0.02


def test_unit_conversion():
    """D_cm2s = D_m2s * 1e4, D_Ang2fs * 1e-5 = D_m2s."""
    data = _make_data()
    res  = calculate_diffusivity(data)
    assert abs(res['D_m2s'] * 1e4 - res['D_cm2s']) < 1e-20
    assert abs(res['D_Ang2fs'] * ANG2_FS_TO_M2_S - res['D_m2s']) < 1e-20


def test_r2_high_on_clean_data():
    data = _make_data(noise=0.0)
    res  = calculate_diffusivity(data)
    assert res['r2'] > 0.999


def test_anisotropy_ratio():
    """Isotropic case -> ratio ~ 1; anisotropic case -> ratio > 5."""
    D = 0.02
    res_iso  = calculate_diffusivity(_make_data(D=D, noise=0.5))
    assert abs(res_iso['anisotropy_ratio'] - 1.0) < 0.15

    # Build anisotropic data manually (D_z >> D_x)
    n = 4001
    steps = np.arange(n) * 500
    t = steps.astype(float)
    data_ani = np.column_stack([
        steps,
        2 * 0.005 * t,   # msd_x
        2 * 0.005 * t,   # msd_y
        2 * 0.05  * t,   # msd_z
        2 * (0.005 + 0.005 + 0.05) * t,  # msd_total
    ])
    res_ani = calculate_diffusivity(data_ani)
    assert res_ani['anisotropy_ratio'] > 5.0


def test_output_keys():
    res = calculate_diffusivity(_make_data())
    for key in ('D_m2s', 'D_cm2s', 'D_Ang2fs', 'slope', 'intercept', 'r2',
                'D_x_m2s', 'D_y_m2s', 'D_z_m2s', 'anisotropy_ratio',
                'time_fs', 'msd_used'):
        assert key in res


def test_detect_diffusive_regime_returns_valid_indices():
    data   = _make_data(n=1001)
    i0, i1 = detect_diffusive_regime(data, timestep_fs=1.0)
    assert isinstance(i0, int) and isinstance(i1, int)
    assert 0 <= i0 < i1 <= len(data)
