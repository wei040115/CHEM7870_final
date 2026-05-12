"""Convert LAMMPS MSD curves into diffusion coefficients."""

import warnings

import numpy as np

from .regression import fit_anisotropic, svd_linear_fit

# slope is Angstrom^2/fs, so D in the same units becomes m^2/s with this factor.
ANG2_FS_TO_M2_S = 1e-5


def _fraction_window(n, fit_start=0.1, fit_end=0.9):
    """Convert fractional fit limits into array indices."""
    return max(1, int(n * fit_start)), min(n, int(n * fit_end))


def _longest_true_run(flags):
    """Find the longest continuous True section in a boolean array."""
    best_start = best_end = 0
    start = None

    for i, flag in enumerate(flags):
        if flag and start is None:
            start = i
        if (not flag or i == len(flags) - 1) and start is not None:
            end = i + 1 if flag else i
            if end - start > best_end - best_start:
                best_start, best_end = start, end
            start = None

    return best_start, best_end


def detect_diffusive_regime(data, timestep_fs=1.0, alpha_min=0.85,
                             alpha_max=1.15, min_points=50):
    """Estimate the part of the MSD curve where log(MSD) vs log(t) has slope ~1."""
    n = len(data)
    fallback = _fraction_window(n)

    t = data[:, 0] * timestep_fs
    msd = data[:, 4]
    valid = (t > 0) & (msd > 0)
    idx = np.flatnonzero(valid)

    if len(idx) < min_points:
        return fallback

    # My note: this auto mode is a check. For the final numbers I still
    # like the fixed 10-90% window because it is easier to explain and compare.
    alpha = np.gradient(np.log(msd[idx]), np.log(t[idx]))
    in_regime = (alpha >= alpha_min) & (alpha <= alpha_max)
    run_start, run_end = _longest_true_run(in_regime)

    if run_end - run_start < min_points:
        warnings.warn(
            'No long diffusive section was found; using the 10-90% window.',
            UserWarning,
            stacklevel=2,
        )
        return fallback

    i0 = int(idx[run_start])
    i1 = int(idx[run_end - 1]) + 1
    return i0, min(n, i1)


def calculate_diffusivity(data, timestep_fs=1.0, fit_start=0.1, fit_end=0.9,
                           auto_detect=False, alpha_min=0.85, alpha_max=1.15):
    """Calculate total and per-axis CO2 self-diffusivity from one MSD array."""
    # I use the fixed window for the main results because it makes every loading
    # comparable. The auto option is mostly a way to check that choice.
    if auto_detect:
        i0, i1 = detect_diffusive_regime(
            data,
            timestep_fs=timestep_fs,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
        )
    else:
        i0, i1 = _fraction_window(len(data), fit_start, fit_end)

    # The fit uses femtoseconds and Angstrom^2, then converts at the end. I keep
    # the simulation units here so it is easier to trace back to the raw file.
    t = data[i0:i1, 0] * timestep_fs
    msd_x = data[i0:i1, 1]
    msd_y = data[i0:i1, 2]
    msd_z = data[i0:i1, 3]
    msd_total = data[i0:i1, 4]

    slope, intercept, r2 = svd_linear_fit(t, msd_total)
    D_ang = slope / 6.0
    D_m2s = D_ang * ANG2_FS_TO_M2_S

    # The per-axis values are a sanity check before trusting the total MSD fit.
    axis = fit_anisotropic(t, msd_x, msd_y, msd_z)
    Dx = axis['D_x'] * ANG2_FS_TO_M2_S
    Dy = axis['D_y'] * ANG2_FS_TO_M2_S
    Dz = axis['D_z'] * ANG2_FS_TO_M2_S

    D_axes = np.array([Dx, Dy, Dz])
    anisotropy_ratio = (
        float(D_axes.max() / D_axes.min()) if D_axes.min() > 0 else float('nan')
    )

    # This is a useful self-check: the average of the 1-D values should be close
    # to the total-D calculation when the fit window is behaving well.
    D_from_axes = (axis['D_x'] + axis['D_y'] + axis['D_z']) / 3.0
    discrepancy = abs(D_from_axes - D_ang) / (abs(D_ang) + 1e-300)
    if discrepancy > 0.05:
        warnings.warn(
            f'Axis-average D differs from total D by {discrepancy:.1%}.',
            UserWarning,
            stacklevel=2,
        )

    return {
        'D_m2s': D_m2s,
        'D_cm2s': D_m2s * 1e4,
        'D_Ang2fs': D_ang,
        'slope': slope,
        'intercept': intercept,
        'r2': r2,
        'D_x_m2s': Dx,
        'D_y_m2s': Dy,
        'D_z_m2s': Dz,
        'slope_x': axis['slope_x'],
        'slope_y': axis['slope_y'],
        'slope_z': axis['slope_z'],
        'intercept_x': axis['intercept_x'],
        'intercept_y': axis['intercept_y'],
        'intercept_z': axis['intercept_z'],
        'r2_x': axis['r2_x'],
        'r2_y': axis['r2_y'],
        'r2_z': axis['r2_z'],
        'anisotropy_ratio': anisotropy_ratio,
        'time_fs': t,
        'msd_used': msd_total,
        'fit_start_idx': i0,
        'fit_end_idx': i1,
    }
