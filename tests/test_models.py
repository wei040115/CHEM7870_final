import numpy as np
import pytest
from diffusivity.models import fit_powerlaw, fit_exponential, fit_linear, compare_models

N = np.array([1000, 1250, 1500, 1750, 2000], dtype=float)


def test_powerlaw_recovers_params():
    a_true, b_true = 1.6e-4, -0.93
    D = a_true * N ** b_true
    res = fit_powerlaw(N, D)
    assert abs(res['a'] - a_true) / a_true < 1e-4
    assert abs(res['b'] - b_true) / abs(b_true) < 1e-4
    assert abs(res['r2'] - 1.0) < 1e-8
    np.testing.assert_allclose(res['predict'](N), D, rtol=1e-4)


def test_exponential_recovers_params():
    D0_true, k_true = 5e-4, 1.5e-3
    D = D0_true * np.exp(-k_true * N)
    res = fit_exponential(N, D)
    assert abs(res['D0'] - D0_true) / D0_true < 1e-4
    assert abs(res['k']  - k_true)  / k_true  < 1e-4
    assert abs(res['r2'] - 1.0) < 1e-8


def test_linear_recovers_params():
    m_true, c_true = -1.2e-10, 4.5e-7
    D = m_true * N + c_true
    res = fit_linear(N, D)
    assert abs(res['slope']     - m_true) / abs(m_true) < 1e-6
    assert abs(res['intercept'] - c_true) / abs(c_true) < 1e-6
    assert abs(res['r2'] - 1.0) < 1e-8


def test_compare_models_ranking_and_delta_aic():
    """compare_models sorts by AIC; best model has delta_aic=0."""
    D = 1.6e-4 * N ** -0.93
    ranked = compare_models(N, D)
    assert len(ranked) == 3
    aics = [m['aic'] for m in ranked]
    assert aics == sorted(aics)
    assert ranked[0]['delta_aic'] == 0.0
    assert all(m['delta_aic'] >= 0 for m in ranked)


def test_powerlaw_wins_on_powerlaw_data():
    D = 1.6e-4 * N ** -0.93
    ranked = compare_models(N, D)
    assert ranked[0]['model_name'] == 'power_law'


def test_all_models_have_predict():
    D = 1.6e-4 * N ** -0.93
    for m in compare_models(N, D):
        assert callable(m['predict'])
