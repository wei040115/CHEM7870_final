"""Save the analysis results in JSON and CSV formats."""

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np


def _json_default(obj):
    """Convert numpy values so `json.dump` can write them."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, np.complexfloating)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')


def _clean_model(model):
    """Drop the predict function because JSON cannot store callables."""
    return {key: value for key, value in model.items() if key != 'predict'}


def results_to_dict(
    n_molecules: list[int],
    msd_results: list[dict],
    uncertainty_results: list[dict] | None = None,
    model_comparison: list[dict] | None = None,
    lit_result: dict | None = None,
    config: dict | None = None,
) -> dict:
    """Put all output pieces into one plain Python dictionary."""
    per_loading = []
    for i, (N, res) in enumerate(zip(n_molecules, msd_results)):
        row = {
            'N_molecules': N,
            'D_m2s': res['D_m2s'],
            'D_cm2s': res['D_cm2s'],
            'D_x_m2s': res.get('D_x_m2s'),
            'D_y_m2s': res.get('D_y_m2s'),
            'D_z_m2s': res.get('D_z_m2s'),
            'anisotropy_ratio': res.get('anisotropy_ratio'),
            'slope': res['slope'],
            'intercept': res['intercept'],
            'r2': res['r2'],
        }

        if uncertainty_results is not None:
            u = uncertainty_results[i]
            row.update({
                'D_block_mean_m2s': u.get('block_D_mean_m2s'),
                'D_block_std_m2s': u.get('block_D_std_m2s'),
            })

        per_loading.append(row)

    models = [_clean_model(model) for model in (model_comparison or [])]
    return {
        'metadata': {
            'generated': datetime.now().isoformat(timespec='seconds'),
            'package': 'diffusivity 0.0.1',
            'config': config,
        },
        'per_loading': per_loading,
        'model_comparison': models,
        'literature': {
            'D_m2s': lit_result['D_m2s'] if lit_result else None,
            'r2': lit_result['r2'] if lit_result else None,
        },
    }


def save_json(results: dict, filepath: str | Path) -> Path:
    """Write one JSON file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, default=_json_default, indent=2)
    print(f'Saved JSON : {path}')
    return path


def _fmt(value, digits=6):
    """Consistent scientific notation for the CSV table."""
    return f'{value:.{digits}e}'


def save_csv(
    n_molecules: list[int],
    msd_results: list[dict],
    uncertainty_results: list[dict] | None = None,
    filepath: str | Path = 'diffusivity_results.csv',
) -> Path:
    """Write the per-loading summary table."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        'N_molecules', 'D_m2s', 'D_cm2s',
        'D_x_m2s', 'D_y_m2s', 'D_z_m2s', 'anisotropy_ratio',
        'r2',
    ]
    if uncertainty_results is not None:
        fieldnames += [
            'D_block_mean_m2s', 'D_block_std_m2s', 'D_block_stderr_m2s',
        ]

    rows = []
    for i, (N, res) in enumerate(zip(n_molecules, msd_results)):
        row = {
            'N_molecules': N,
            'D_m2s': _fmt(res['D_m2s']),
            'D_cm2s': _fmt(res['D_cm2s']),
            'D_x_m2s': _fmt(res.get('D_x_m2s', float('nan'))),
            'D_y_m2s': _fmt(res.get('D_y_m2s', float('nan'))),
            'D_z_m2s': _fmt(res.get('D_z_m2s', float('nan'))),
            'anisotropy_ratio': f'{res.get("anisotropy_ratio", float("nan")):.4f}',
            'r2': f'{res["r2"]:.6f}',
        }

        if uncertainty_results is not None:
            u = uncertainty_results[i]
            row.update({
                'D_block_mean_m2s': _fmt(u.get('block_D_mean_m2s', float('nan'))),
                'D_block_std_m2s': _fmt(u.get('block_D_std_m2s', float('nan'))),
                'D_block_stderr_m2s': _fmt(u.get('block_D_stderr_m2s', float('nan'))),
            })

        rows.append(row)

    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'Saved CSV  : {path}')
    return path


def save_all(
    n_molecules: list[int],
    msd_results: list[dict],
    uncertainty_results: list[dict] | None = None,
    model_comparison: list[dict] | None = None,
    lit_result: dict | None = None,
    config: dict | None = None,
    out_dir: str | Path = '.',
    prefix: str = 'diffusivity',
) -> tuple[Path, Path]:
    """Write both JSON and CSV in one call."""
    out_dir = Path(out_dir)
    results = results_to_dict(
        n_molecules,
        msd_results,
        uncertainty_results,
        model_comparison,
        lit_result,
        config,
    )
    json_path = save_json(results, out_dir / f'{prefix}_results.json')
    csv_path = save_csv(
        n_molecules,
        msd_results,
        uncertainty_results,
        out_dir / f'{prefix}_results.csv',
    )
    return json_path, csv_path
