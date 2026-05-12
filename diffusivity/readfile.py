"""Small helpers for reading the LAMMPS MSD files used in this project."""

import re
import warnings
from pathlib import Path

import numpy as np


def load_msd_file(filepath: str | Path) -> np.ndarray:
    """Read one MSD file and do the basic checks I need before fitting."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f'MSD file not found: {path}')

    data = np.loadtxt(path, comments='#')
    data = np.atleast_2d(data)

    if data.shape[1] < 5:
        raise ValueError(
            f'{path.name}: expected at least 5 columns '
            f'(step, MSD_x, MSD_y, MSD_z, MSD_total), got {data.shape[1]}'
        )

    # I keep these as warnings because a short or slightly messy file can still
    # be useful for a quick check, but I would not trust it for final numbers.
    if data.shape[0] < 10:
        warnings.warn(
            f'{path.name}: only {data.shape[0]} data rows; fit may be unreliable.',
            UserWarning,
            stacklevel=2,
        )

    if np.any(np.diff(data[:, 0]) <= 0):
        warnings.warn(
            f'{path.name}: step column is not strictly increasing.',
            UserWarning,
            stacklevel=2,
        )

    return data


def load_loading_series(
    data_dir: str | Path,
    pattern: str = r'msd_co2_(\d+)\.dat',
    exclude: list[str] | None = None,
) -> tuple[list[int], list[np.ndarray]]:
    """Load every `msd_co2_<N>.dat` file and sort by molecule count."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f'Data directory not found: {data_dir}')

    skip = set(exclude or [])
    pairs: list[tuple[int, np.ndarray]] = []

    for path in sorted(data_dir.iterdir()):
        if path.name in skip:
            continue

        # The molecule count is in the filename, so this keeps the data files
        # and the loading labels tied together.
        match = re.fullmatch(pattern, path.name)
        if match is None:
            continue

        pairs.append((int(match.group(1)), load_msd_file(path)))

    if not pairs:
        raise RuntimeError(f'No files matching "{pattern}" found in {data_dir}')

    pairs.sort(key=lambda item: item[0])
    n_molecules, datasets = zip(*pairs)
    return list(n_molecules), list(datasets)


def load_literature(filepath: str | Path) -> np.ndarray:
    """Same file format as my simulation MSD files, just kept separate."""
    return load_msd_file(filepath)
