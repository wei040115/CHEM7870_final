"""Simple D(N) models for checking the loading trend."""

import numpy as np

from .regression import svd_linear_fit


def _arrays(n_molecules, diffusivities):
    """Use float arrays so the math below is predictable."""
    return np.asarray(n_molecules, dtype=float), np.asarray(diffusivities, dtype=float)


def _aic(n, rss, k=2):
    """AIC for models with the same number of fit parameters."""
    return n * np.log(rss / n + 1e-300) + 2 * k


def _rss(D, D_pred):
    """Residual sum of squares in the original D units."""
    return float(np.sum((D - D_pred) ** 2))


def fit_powerlaw(n_molecules, diffusivities):
    """Fit D = a*N^b by fitting ln(D) vs ln(N)."""
    N, D = _arrays(n_molecules, diffusivities)

    slope, intercept, r2 = svd_linear_fit(np.log(N), np.log(D))
    a = float(np.exp(intercept))
    b = float(slope)

    def predict(N_new):
        return a * np.asarray(N_new, dtype=float) ** b

    # My note: I compare RSS after converting back to D, so all models are
    # judged on the same scale as the plotted diffusivity values.
    rss = _rss(D, predict(N))
    return {
        'model_name': 'power_law',
        'label': f'Power law  D~N^{b:.3f}',
        'a': a,
        'b': b,
        'params': {'a': a, 'b': b},
        'r2': r2,
        'aic': _aic(len(N), rss),
        'predict': predict,
    }


def fit_exponential(n_molecules, diffusivities):
    """Fit D = D0*exp(-k*N) by fitting ln(D) vs N."""
    N, D = _arrays(n_molecules, diffusivities)

    slope, intercept, r2 = svd_linear_fit(N, np.log(D))
    D0 = float(np.exp(intercept))
    k = float(-slope)

    def predict(N_new):
        return D0 * np.exp(-k * np.asarray(N_new, dtype=float))

    rss = _rss(D, predict(N))
    return {
        'model_name': 'exponential',
        'label': f'Exponential  D0={D0:.3e}, k={k:.3e}',
        'D0': D0,
        'k': k,
        'params': {'D0': D0, 'k': k},
        'r2': r2,
        'aic': _aic(len(N), rss),
        'predict': predict,
    }


def fit_linear(n_molecules, diffusivities):
    """Fit the straight-line model D = m*N + c."""
    N, D = _arrays(n_molecules, diffusivities)

    slope, intercept, r2 = svd_linear_fit(N, D)

    def predict(N_new):
        return slope * np.asarray(N_new, dtype=float) + intercept

    rss = _rss(D, predict(N))
    return {
        'model_name': 'linear',
        'label': f'Linear  m={slope:.3e}',
        'slope': slope,
        'intercept': intercept,
        'params': {'slope': slope, 'intercept': intercept},
        'r2': r2,
        'aic': _aic(len(N), rss),
        'predict': predict,
    }


def compare_models(n_molecules, diffusivities):
    """Fit the three models and rank them from lowest to highest AIC."""
    # With only five loading points, I kept the candidate models simple. A more
    # flexible curve would probably fit better but would be harder to interpret.
    fits = [
        fit_powerlaw(n_molecules, diffusivities),
        fit_exponential(n_molecules, diffusivities),
        fit_linear(n_molecules, diffusivities),
    ]
    fits.sort(key=lambda model: model['aic'])

    best_aic = fits[0]['aic']
    for model in fits:
        model['delta_aic'] = model['aic'] - best_aic

    return fits
