import numpy as np
import pytest
from diffusivity.uncertainty import block_average_diffusivity, combined_uncertainty


def _make_data(slope=0.15, n=4001, noise=20.0, seed=0):
    rng   = np.random.default_rng(seed)
    steps = np.arange(n) * 500
    msd   = slope * steps.astype(float) + rng.normal(0, noise, n)
    msd   = np.clip(msd, 0, None)
    dummy = msd / 3
    return np.column_stack([steps, dummy, dummy, dummy, msd])


def test_block_average_D_positive_and_ci_ordered():
    res = block_average_diffusivity(_make_data(), n_blocks=5)
    assert res['D_mean_m2s'] > 0
    assert res['ci_68_lo'] < res['D_mean_m2s'] < res['ci_68_hi']
    assert res['n_blocks_used'] == 5


def test_combined_uncertainty_has_prefixed_keys():
    res = combined_uncertainty(_make_data(), n_blocks=5)
    for key in ('block_D_mean_m2s', 'block_D_stderr_m2s'):
        assert key in res
