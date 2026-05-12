"""Uncertainty estimates for the fitted diffusivity values."""

import numpy as np

from .regression import svd_linear_fit

ANG2_FS_TO_M2_S = 1e-5


def _get_window(data, fit_start, fit_end):
    """Use the same fractional window as the main MSD fit."""
    n = len(data)
    return max(1, int(n * fit_start)), int(n * fit_end)


def _fit_D_from_window(t, msd):
    """Fit total MSD and return D in Angstrom^2/fs."""
    slope, _, _ = svd_linear_fit(t, msd)
    return slope / 6.0


def block_average_diffusivity(data, n_blocks=5, timestep_fs=1.0,
                               fit_start=0.1, fit_end=0.9):
    """Split the fit window into blocks and fit D once per block."""
    i0, i1 = _get_window(data, fit_start, fit_end)
    blocks = np.array_split(np.arange(i0, i1), n_blocks)

    D_blocks = []
    for block in blocks:
        if len(block) < 4:
            continue

        # My note: each block still has to be long enough to look roughly
        # linear; otherwise the standard error is not very meaningful.
        # This is why I use a small number of blocks instead of many tiny ones.
        t = data[block, 0] * timestep_fs
        msd = data[block, 4]
        D_blocks.append(_fit_D_from_window(t, msd))

    D_ang = np.asarray(D_blocks)
    D_m2s = D_ang * ANG2_FS_TO_M2_S
    mean = float(D_m2s.mean())
    std = float(D_m2s.std(ddof=1)) if len(D_m2s) > 1 else 0.0
    stderr = std / np.sqrt(len(D_m2s))

    return {
        'D_mean_m2s': mean,
        'D_std_m2s': std,
        'D_stderr_m2s': stderr,
        'D_blocks_m2s': D_m2s,
        'D_blocks_Ang2fs': D_ang,
        'n_blocks_used': len(D_m2s),
        'ci_68_lo': mean - stderr,
        'ci_68_hi': mean + stderr,
    }


def combined_uncertainty(data, timestep_fs=1.0, fit_start=0.1, fit_end=0.9,
                          n_blocks=5):
    """Run block averaging and return results with block_ prefix."""
    block = block_average_diffusivity(
        data,
        n_blocks=n_blocks,
        timestep_fs=timestep_fs,
        fit_start=fit_start,
        fit_end=fit_end,
    )
    return {f'block_{key}': value for key, value in block.items()}
