"""Plotting functions for the final project figures."""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

_COLORS = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0']


def _save(fig: plt.Figure, path: str | Path) -> Path:
    """Save and close a figure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')
    return path


def _grid(n_plots, ncols=3, width=5, height=4):
    """Make a simple grid of subplots and always return a flat axes array."""
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(width * ncols, height * nrows))
    return fig, np.atleast_1d(axes).ravel()


def _hide_extra_axes(axes, n_used):
    """Hide unused panels in the last row."""
    for ax in axes[n_used:]:
        ax.set_visible(False)


def plot_msd_fits(
    n_molecules: list[int],
    datasets: list[np.ndarray],
    results: list[dict],
    timestep_fs: float = 1.0,
    output: str | Path = 'msd_fits.png',
) -> Path:
    """Plot total MSD curves with the linear fit used for D."""
    fig, axes = _grid(len(n_molecules), ncols=3)
    colors = cm.viridis(np.linspace(0.15, 0.85, len(n_molecules)))

    for ax, N, data, result, color in zip(axes, n_molecules, datasets, results, colors):
        t_ps = data[:, 0] * timestep_fs / 1e3
        t_fit_ps = result['time_fs'] / 1e3
        msd_fit = result['slope'] * result['time_fs'] + result['intercept']

        ax.plot(t_ps, data[:, 4], '.', ms=2, color=color, alpha=0.4, label='data')
        ax.plot(t_fit_ps, msd_fit, 'k-', lw=1.5,
                label=f'fit  R2={result["r2"]:.4f}')
        ax.set_title(f'N = {N}\nD = {result["D_m2s"]:.3e} m2/s')
        ax.set_xlabel('Time (ps)')
        ax.set_ylabel('MSD (Angstrom^2)')
        ax.legend(fontsize=7)

    _hide_extra_axes(axes, len(n_molecules))
    fig.suptitle('MSD fits used for diffusivity', fontsize=13, y=1.01)
    fig.tight_layout()
    return _save(fig, output)


def plot_msd_anisotropic(
    n_molecules: list[int],
    datasets: list[np.ndarray],
    results: list[dict],
    timestep_fs: float = 1.0,
    output: str | Path = 'msd_anisotropic.png',
) -> Path:
    """Plot x/y/z MSD curves to see whether one direction behaves differently."""
    fig, axes = _grid(len(n_molecules), ncols=min(len(n_molecules), 3))
    colors = {'x': '#E53935', 'y': '#43A047', 'z': '#1E88E5', 'total': 'black'}
    columns = [(1, 'x'), (2, 'y'), (3, 'z'), (4, 'total')]

    for ax, N, data, result in zip(axes, n_molecules, datasets, results):
        t_ps = data[:, 0] * timestep_fs / 1e3
        for col, label in columns:
            ax.plot(t_ps, data[:, col], '.', ms=2, color=colors[label], alpha=0.3)

        t_fit_ps = result['time_fs'] / 1e3
        for label in ('x', 'y', 'z'):
            fit = result[f'slope_{label}'] * result['time_fs'] + result[f'intercept_{label}']
            ax.plot(t_fit_ps, fit, '-', lw=1.5, color=colors[label], label=f'MSD_{label}')

        total_fit = result['slope'] * result['time_fs'] + result['intercept']
        ax.plot(t_fit_ps, total_fit, '-', lw=1.5, color=colors['total'], label='MSD_total')

        ax.set_title(f'N = {N} | anisotropy = {result["anisotropy_ratio"]:.2f}')
        ax.set_xlabel('Time (ps)')
        ax.set_ylabel('MSD (Angstrom^2)')
        ax.legend(fontsize=7, ncol=2)

    _hide_extra_axes(axes, len(n_molecules))
    fig.suptitle('MSD components by direction', fontsize=13, y=1.01)
    fig.tight_layout()
    return _save(fig, output)


def plot_D_anisotropic(
    n_molecules: list[int],
    results: list[dict],
    output: str | Path = 'D_anisotropic.png',
) -> Path:
    """Bar chart of Dx, Dy, and Dz for each loading."""
    x = np.arange(len(n_molecules))
    width = 0.25
    Dx = np.array([r['D_x_m2s'] for r in results]) * 1e7
    Dy = np.array([r['D_y_m2s'] for r in results]) * 1e7
    Dz = np.array([r['D_z_m2s'] for r in results]) * 1e7

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, Dx, width, label='D_x', color='#E53935', alpha=0.8)
    ax.bar(x, Dy, width, label='D_y', color='#43A047', alpha=0.8)
    ax.bar(x + width, Dz, width, label='D_z', color='#1E88E5', alpha=0.8)

    # I scale by 1e7 only to make the y-axis numbers easier to read.
    ax.set_xticks(x)
    ax.set_xticklabels([str(N) for N in n_molecules])
    ax.set_xlabel('Number of CO2 molecules')
    ax.set_ylabel('Diffusivity (x10^-7 m2/s)')
    ax.set_title('Directional diffusivity components')
    ax.legend()
    fig.tight_layout()
    return _save(fig, output)


def _draw_D_vs_N(
    ax: plt.Axes,
    N: np.ndarray,
    D: np.ndarray,
    model_comparison: list[dict],
    N_dense: np.ndarray,
    lit_N: int | None,
    lit_D: float | None,
    xscale: str,
    yscale: str,
) -> None:
    """Draw the shared parts of the linear and log-log D(N) plots."""
    styles = ['-', '--', ':']
    for i, model in enumerate(model_comparison):
        label = (
            f'{model["label"]}  '
            f'(AIC={model["aic"]:.1f}, dAIC={model["delta_aic"]:.1f})'
        )
        ax.plot(N_dense, model['predict'](N_dense), styles[i % len(styles)],
                lw=2, color=_COLORS[i], label=label)

    ax.scatter(N, D, color='black', s=70, zorder=5, label='MD fit')
    if lit_N is not None and lit_D is not None:
        ax.scatter([lit_N], [lit_D], marker='*', s=220,
                   color='gold', edgecolors='k', zorder=6, label='Literature')

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlabel('Number of CO2 molecules')
    ax.set_ylabel('D (m2/s)')


def plot_D_vs_N(
    n_molecules: list[int],
    diffusivities: list[float],
    model_comparison: list[dict],
    lit_N: int | None = None,
    lit_D: float | None = None,
    output: str | Path = 'D_vs_N.png',
) -> Path:
    """Plot D(N) on normal axes."""
    N = np.asarray(n_molecules, dtype=float)
    D = np.asarray(diffusivities, dtype=float)
    N_dense = np.linspace(N.min() * 0.8, N.max() * 1.2, 300)

    fig, ax = plt.subplots(figsize=(9, 6))
    _draw_D_vs_N(ax, N, D, model_comparison, N_dense, lit_N, lit_D,
                 'linear', 'linear')
    ax.set_title('CO2 diffusivity vs loading')
    ax.legend(fontsize=8, loc='upper right')
    fig.tight_layout()
    return _save(fig, output)


def plot_D_vs_N_loglog(
    n_molecules: list[int],
    diffusivities: list[float],
    model_comparison: list[dict],
    lit_N: int | None = None,
    lit_D: float | None = None,
    output: str | Path = 'D_vs_N_loglog.png',
) -> Path:
    """Plot D(N) on log-log axes, which is useful for the power-law model."""
    N = np.asarray(n_molecules, dtype=float)
    D = np.asarray(diffusivities, dtype=float)
    N_dense = np.linspace(N.min() * 0.8, N.max() * 1.2, 300)

    fig, ax = plt.subplots(figsize=(9, 6))
    _draw_D_vs_N(ax, N, D, model_comparison, N_dense, lit_N, lit_D,
                 'log', 'log')
    ax.set_title('CO2 diffusivity vs loading (log-log)')
    ax.legend(fontsize=8, loc='upper right')
    fig.tight_layout()
    return _save(fig, output)




def plot_all(
    n_molecules: list[int],
    datasets: list[np.ndarray],
    msd_results: list[dict],
    model_comparison: list[dict],
    lit_N: int | None = None,
    lit_D: float | None = None,
    timestep_fs: float = 1.0,
    out_dir: str | Path = '.',
    prefix: str = 'diffusivity',
) -> list[Path]:
    """Generate all figures for the project."""
    out = Path(out_dir)
    D_values = [result['D_m2s'] for result in msd_results]

    saved = [
        plot_msd_fits(
            n_molecules, datasets, msd_results, timestep_fs,
            out / f'{prefix}_msd_fits.png',
        ),
        plot_msd_anisotropic(
            n_molecules, datasets, msd_results, timestep_fs,
            out / f'{prefix}_msd_anisotropic.png',
        ),
        plot_D_anisotropic(
            n_molecules, msd_results,
            out / f'{prefix}_D_anisotropic.png',
        ),
        plot_D_vs_N(
            n_molecules, D_values, model_comparison, lit_N, lit_D,
            out / f'{prefix}_D_vs_N.png',
        ),
        plot_D_vs_N_loglog(
            n_molecules, D_values, model_comparison, lit_N, lit_D,
            out / f'{prefix}_D_vs_N_loglog.png',
        ),
    ]
    return saved
