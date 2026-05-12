"""Command-line driver for the full CO2 loading-series analysis."""

import argparse
from pathlib import Path

DATA_DIR = 'data'
LIT_FILE = 'msd_co2_literature.dat'
TIMESTEP_FS = 2.0
FIT_START = 0.10
FIT_END = 0.90
N_BLOCKS = 5
OUT_DIR = 'output'
PREFIX = 'diffusivity'


def _section(title):
    """Print a small divider so the command-line output is easier to scan."""
    print('\n' + '-' * 60)
    print(f'  {title}')
    print('-' * 60)


def _record_run_settings(args, lit_path):
    """Record the settings I used for this run in the JSON output."""
    return {
        'data': {
            'dir': str(args.data_dir),
            'pattern': r'msd_co2_(\d+)\.dat',
            'literature_file': str(lit_path),
        },
        'simulation': {'timestep_fs': args.timestep},
        'fitting': {
            'fit_start': args.fit_start,
            'fit_end': args.fit_end,
            'auto_detect': args.auto_detect,
        },
        'uncertainty': {
            'n_blocks': args.blocks,
        },
        'output': {
            'dir': str(args.out_dir),
            'prefix': args.prefix,
            'json': not args.no_json,
            'csv': not args.no_csv,
            'figures': not args.no_plots,
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='diffusivity',
        description='CO2 self-diffusivity from LAMMPS MSD trajectories.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--data-dir', default=DATA_DIR, type=Path)
    parser.add_argument('--timestep', default=TIMESTEP_FS, type=float, metavar='FS')
    parser.add_argument('--fit-start', default=FIT_START, type=float)
    parser.add_argument('--fit-end', default=FIT_END, type=float)
    parser.add_argument('--auto-detect', action='store_true')
    parser.add_argument('--select', nargs='+', type=int, metavar='N',
                        help='Only analyse these molecule counts, e.g. --select 1000 2000')
    parser.add_argument('--blocks', default=N_BLOCKS, type=int, metavar='N')
    parser.add_argument('--no-uncertainty', action='store_true')
    parser.add_argument('--out-dir', default=OUT_DIR, type=Path)
    parser.add_argument('--prefix', default=PREFIX)
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--no-json', action='store_true')
    parser.add_argument('--no-csv', action='store_true')
    args = parser.parse_args(argv)

    import diffusivity as diff
    from diffusivity.export import results_to_dict

    out_dir = Path(args.out_dir) / args.prefix
    out_dir.mkdir(parents=True, exist_ok=True)

    _section('Loading MSD data')
    n_molecules, datasets = diff.load_loading_series(args.data_dir, exclude=[LIT_FILE])
    if args.select:
        # Useful while checking plots: I can rerun only one or two loadings.
        selected = set(args.select)
        pairs = [(n, d) for n, d in zip(n_molecules, datasets) if n in selected]
        if not pairs:
            raise SystemExit(f'No files found for --select {args.select}')
        n_molecules, datasets = zip(*pairs)
        n_molecules, datasets = list(n_molecules), list(datasets)
    print(f'  {len(n_molecules)} files  |  N = {n_molecules}')

    lit_result = None
    lit_path = Path(args.data_dir) / LIT_FILE
    if lit_path.exists():
        # I keep the literature MSD separate from my loading trend, since it is
        # a reference point rather than one of the five simulation sizes.
        lit_data = diff.load_literature(lit_path)
        lit_result = diff.calculate_diffusivity(
            lit_data,
            timestep_fs=args.timestep,
            fit_start=args.fit_start,
            fit_end=args.fit_end,
            auto_detect=args.auto_detect,
        )
        print(f'  Literature file: D = {lit_result["D_m2s"]:.4e} m2/s')

    _section('SVD linear regression  (MSD = 6Dt + b)')
    msd_results = []
    for N, data in zip(n_molecules, datasets):
        result = diff.calculate_diffusivity(
            data,
            timestep_fs=args.timestep,
            fit_start=args.fit_start,
            fit_end=args.fit_end,
            auto_detect=args.auto_detect,
        )
        msd_results.append(result)
        print(
            f'  N={N:5d}  D={result["D_m2s"]:9.3e} m2/s  '
            f'Dx={result["D_x_m2s"]:.3e}  Dy={result["D_y_m2s"]:.3e}  '
            f'Dz={result["D_z_m2s"]:.3e}  R2={result["r2"]:.5f}'
        )

    uncertainty_results = None
    if not args.no_uncertainty:
        _section('Uncertainty quantification')
        uncertainty_results = []
        for N, data in zip(n_molecules, datasets):
            u = diff.combined_uncertainty(
                data,
                timestep_fs=args.timestep,
                fit_start=args.fit_start,
                fit_end=args.fit_end,
                n_blocks=args.blocks,
            )
            uncertainty_results.append(u)
            print(
                f'  N={N:5d}  block: {u["block_D_mean_m2s"]:.3e} '
                f'+/- {u["block_D_stderr_m2s"]:.2e}'
            )

    _section('D(N) model comparison')
    D_list = [result['D_m2s'] for result in msd_results]
    ranked = diff.compare_models(n_molecules, D_list)
    print(f'  {"Model":15s}  {"AIC":>9}  {"dAIC":>7}  {"R2":>8}')
    for model in ranked:
        print(
            f'  {model["model_name"]:15s}  {model["aic"]:9.2f}  '
            f'{model["delta_aic"]:7.2f}  {model["r2"]:8.5f}'
        )
    print(f'\n  Best model: {ranked[0]["model_name"]}')

    if not args.no_plots:
        _section('Generating figures')
        saved = diff.plot.plot_all(
            n_molecules,
            datasets,
            msd_results,
            ranked,
            lit_D=lit_result['D_m2s'] if lit_result else None,
            timestep_fs=args.timestep,
            out_dir=out_dir,
            prefix=args.prefix,
        )
        print(f'  {len(saved)} figures saved to {out_dir}')

    if not args.no_json or not args.no_csv:
        _section('Exporting results')
        config = _record_run_settings(args, lit_path)

        if not args.no_json:
            results = results_to_dict(
                n_molecules,
                msd_results,
                uncertainty_results,
                ranked,
                lit_result=lit_result,
                config=config,
            )
            diff.save_json(results, out_dir / f'{args.prefix}_results.json')

        if not args.no_csv:
            diff.save_csv(
                n_molecules,
                msd_results,
                uncertainty_results,
                out_dir / f'{args.prefix}_results.csv',
            )

    _section(f'Done. Output directory: {out_dir.resolve()}')


if __name__ == '__main__':
    main()
