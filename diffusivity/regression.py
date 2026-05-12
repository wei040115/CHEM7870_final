"""Linear fits for MSD data.

I use SVD instead of the normal equations because the time column can be large
for MD output. This keeps the regression step stable without changing the
straight-line model.
"""

import numpy as np


def _svd_fit_full(t, y):
    """Return the line fit plus the arrays needed for diagnostics."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    A = np.column_stack([t, np.ones_like(t)])
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    coeffs = Vt.T @ (U.T @ y / s)

    y_pred = A @ coeffs
    residuals = y - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    slope, intercept = coeffs
    return float(slope), float(intercept), float(r2), residuals, y_pred


def svd_linear_fit(t, y):
    """Fit y = slope*t + intercept and return slope, intercept, and R2."""
    slope, intercept, r2, _, _ = _svd_fit_full(t, y)
    return slope, intercept, r2


def svd_linear_fit_with_residuals(t, y):
    """Same fit as above, but also keep residuals for plotting/checking."""
    return _svd_fit_full(t, y)


def fit_anisotropic(t, msd_x, msd_y, msd_z):
    """Fit the x, y, and z MSD curves separately."""
    result = {}
    for axis, msd in {'x': msd_x, 'y': msd_y, 'z': msd_z}.items():
        slope, intercept, r2 = svd_linear_fit(t, msd)

        # My note: for one coordinate the Einstein factor is 2, not 6.
        result[f'D_{axis}'] = slope / 2.0
        result[f'slope_{axis}'] = slope
        result[f'intercept_{axis}'] = intercept
        result[f'r2_{axis}'] = r2

    return result
