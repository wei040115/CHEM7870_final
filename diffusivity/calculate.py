"""Single-file analysis: compute D from one MSD file and save all outputs."""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path


def calculate_single(
    filepath: str | Path,
    out_dir: str | Path = 'output',
    timestep_fs: float = 2.0,
    fit_start: float = 0.1,
    fit_end: float = 0.9,
    auto_detect: bool = False,
    no_plot: bool = False,
    no_json: bool = False,
    no_csv: bool = False,
) -> dict:
    """Compute diffusivity from one .dat file and write all outputs to *out_dir*.

    Returns the result dict from calculate_diffusivity.
    """
    from diffusivity.readfile import load_msd_file
    from diffusivity.msd import calculate_diffusivity
    from diffusivity.export import save_json

    filepath = Path(filepath)
    stem = filepath.stem  # e.g. msd_co2_1000
    out_dir = Path(out_dir) / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_msd_file(filepath)
    result = calculate_diffusivity(
        data,
        timestep_fs=timestep_fs,
        fit_start=fit_start,
        fit_end=fit_end,
        auto_detect=auto_detect,
    )

    if not no_plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        t_all_ps = data[:, 0] * timestep_fs / 1e3
        msd_all = data[:, 4]
        t_fit_ps = result['time_fs'] / 1e3
        msd_fit = result['slope'] * result['time_fs'] + result['intercept']

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(t_all_ps, msd_all, '.', ms=2, color='steelblue',
                alpha=0.45, label='MSD data')
        ax.plot(t_fit_ps, msd_fit, color='crimson', lw=2,
                label=f'SVD fit  R2 = {result["r2"]:.4f}')
        ax.text(
            0.04, 0.96,
            f'D = {result["D_m2s"]:.4e} m2/s\n'
            f'D = {result["D_cm2s"]:.4e} cm2/s',
            transform=ax.transAxes, fontsize=12, va='top', family='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                      edgecolor='#aaa', alpha=0.9),
        )
        ax.set_xlabel('Time (ps)')
        ax.set_ylabel('MSD (Angstrom^2)')
        ax.set_title(f'{filepath.name}  —  MSD linear fit')
        ax.legend()
        fig.tight_layout()

        plot_path = out_dir / f'{stem}_msd_fit.png'
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved plot : {plot_path}')

    if not no_json:
        record = {
            'metadata': {
                'generated': datetime.now().isoformat(timespec='seconds'),
                'source_file': str(filepath),
                'timestep_fs': timestep_fs,
                'fit_start': fit_start,
                'fit_end': fit_end,
                'auto_detect': auto_detect,
            },
            'D_m2s': result['D_m2s'],
            'D_cm2s': result['D_cm2s'],
            'D_x_m2s': result['D_x_m2s'],
            'D_y_m2s': result['D_y_m2s'],
            'D_z_m2s': result['D_z_m2s'],
            'anisotropy_ratio': result['anisotropy_ratio'],
            'slope': result['slope'],
            'intercept': result['intercept'],
            'r2': result['r2'],
        }
        save_json(record, out_dir / f'{stem}_result.json')

    if not no_csv:
        csv_path = out_dir / f'{stem}_result.csv'
        fieldnames = [
            'source_file', 'D_m2s', 'D_cm2s',
            'D_x_m2s', 'D_y_m2s', 'D_z_m2s',
            'anisotropy_ratio', 'r2',
        ]
        row = {
            'source_file': str(filepath),
            'D_m2s': f'{result["D_m2s"]:.6e}',
            'D_cm2s': f'{result["D_cm2s"]:.6e}',
            'D_x_m2s': f'{result["D_x_m2s"]:.6e}',
            'D_y_m2s': f'{result["D_y_m2s"]:.6e}',
            'D_z_m2s': f'{result["D_z_m2s"]:.6e}',
            'anisotropy_ratio': f'{result["anisotropy_ratio"]:.4f}',
            'r2': f'{result["r2"]:.6f}',
        }
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)
        print(f'Saved CSV  : {csv_path}')

    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog='calculate',
        description='Compute diffusivity from one LAMMPS MSD file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('file', type=Path)
    parser.add_argument('--timestep', type=float, default=2.0, metavar='FS')
    parser.add_argument('--fit-start', type=float, default=0.1, metavar='FRAC')
    parser.add_argument('--fit-end', type=float, default=0.9, metavar='FRAC')
    parser.add_argument('--auto-detect', action='store_true')
    parser.add_argument('--out-dir', type=Path, default=Path('output'),
                        help='Directory for all output files')
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--no-json', action='store_true')
    parser.add_argument('--no-csv', action='store_true')
    args = parser.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).parents[1]))

    if not args.file.exists():
        sys.exit(f'Error: file not found - {args.file}')

    result = calculate_single(
        filepath=args.file,
        out_dir=args.out_dir,
        timestep_fs=args.timestep,
        fit_start=args.fit_start,
        fit_end=args.fit_end,
        auto_detect=args.auto_detect,
        no_plot=args.no_plot,
        no_json=args.no_json,
        no_csv=args.no_csv,
    )

    print(f'\nFile  : {args.file}')
    print(f'D     = {result["D_m2s"]:.4e} m2/s  ({result["D_cm2s"]:.4e} cm2/s)')
    print(f'R2    = {result["r2"]:.5f}')
    print(f'Out   : {args.out_dir.resolve()}')


if __name__ == '__main__':
    main()
