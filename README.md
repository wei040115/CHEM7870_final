# CHEM 7870 final project: CO2 diffusivity

This project calculates CO2 self-diffusivity from LAMMPS mean-squared
displacement (MSD) trajectories. The main question I use it for is how the
diffusion coefficient changes as the number of CO2 molecules in the simulation
box increases from 1000 to 2000.

The default workflow fits the total MSD with the 3-D Einstein relation, reports
per-axis diffusivities as a check for anisotropy, estimates uncertainty with
block averaging, and compares simple D(N) models.

## Installation

```bash
pip install -e .
```

For running the tests:

```bash
pip install -e ".[dev]"
```

Runtime dependencies are `numpy` and `matplotlib`; `pytest` is only needed for
the test suite.

## Quick start

```bash
# Analyze one MSD file (outputs go into output/msd_co2_1000/)
python -m diffusivity.calculate data/msd_co2_1000.dat

# Run the full loading series (outputs go into output/diffusivity/)
python -m diffusivity

# Only analyze a subset of loadings
python -m diffusivity --select 1000 2000

# Skip uncertainty quantification
python -m diffusivity --no-uncertainty

# Send outputs to another directory
python -m diffusivity --out-dir results/

# Skip selected outputs
python -m diffusivity --no-plots --no-json

#calculate the single CO2 diffusivity from data folder
calculate data/[file_name]
```
for example, to calculate the loading of 1000 number of CO2
```bash
calculate data/msd_co2_1000
```


The full run reads all `data/msd_co2_<N>.dat` files, excluding
`data/msd_co2_literature.dat`. The literature file is treated as a separate
reference point and is not used in the D(N) model fits.

## Data

Input files are LAMMPS MSD text files with two comment lines and five numeric
columns:

```text
# step  MSD_x(Ang2)  MSD_y  MSD_z  MSD_total
# TimeStep c_msd_co2[1] c_msd_co2[2] c_msd_co2[3] c_msd_co2[4]
0    0.0   0.0   0.0   0.0
500  ...
```

Steps are converted to time with the default `timestep_fs = 2.0`, and MSD values
are in Angstrom squared.

Current loading series:

| N molecules | D (m2/s) | R2 |
|---:|---:|---:|
| 1000 | 1.263e-07 | 0.99845 |
| 1250 | 1.027e-07 | 0.99967 |
| 1500 | 8.348e-08 | 0.99949 |
| 1750 | 7.733e-08 | 0.99979 |
| 2000 | 6.506e-08 | 0.99918 |

## Analysis choices

Default parameters are defined near the top of `diffusivity/__main__.py` and can
be changed from the command line.

| Parameter | Default | Why it is used |
|---|---:|---|
| `timestep_fs` | 2.0 | Converts LAMMPS steps to femtoseconds |
| `fit_start` | 0.10 | Leaves out the early ballistic part of the MSD curve |
| `fit_end` | 0.90 | Avoids putting too much weight on the noisiest tail |
| `n_blocks` | 5 | Splits each trajectory into block fits for uncertainty |

Example:

```bash
python -m diffusivity --fit-start 0.15 --fit-end 0.85 --blocks 10
```

There is also an experimental `--auto-detect` option that looks for a log-log
MSD slope close to 1 before fitting.

## Notes from my analysis

The main trend I see is that diffusivity decreases as the CO2 loading increases.
That matches the physical picture I expected: at higher loading, molecules have
less open space to move through, so the MSD slope becomes smaller.

I kept the 10%-90% fit window as the default because it is easy to justify when
looking at the MSD curves. The first part can include early-time motion that is
not fully diffusive, and the very end can be noisier. The automatic window
finder is useful as a check, but I did not want the final table to depend on a
hidden fitting choice.

The per-axis values are mainly a warning check for me. If Dx, Dy, and Dz were
very different, I would be less comfortable reporting only one isotropic D
value.

## Package structure

| Module | Purpose |
|---|---|
| `readfile.py` | Load and validate MSD files |
| `regression.py` | SVD linear regression used for MSD fitting |
| `msd.py` | Einstein-relation diffusivity calculation |
| `uncertainty.py` | Block averaging uncertainty estimate |
| `models.py` | Power-law, exponential, and linear D(N) model fits |
| `plot.py` | Figure generation |
| `export.py` | JSON and CSV output |
| `calculate.py` | Single-file command-line entry point |
| `__main__.py` | Full loading-series command-line entry point |

## Outputs

`python -m diffusivity` writes to `output/diffusivity/`:

- `diffusivity_results.json`
- `diffusivity_results.csv`
- `diffusivity_msd_fits.png`
- `diffusivity_msd_anisotropic.png`
- `diffusivity_D_anisotropic.png`
- `diffusivity_D_vs_N.png`
- `diffusivity_D_vs_N_loglog.png`

`python -m diffusivity.calculate data/msd_co2_1000.dat` writes to
`output/msd_co2_1000/`:

- `msd_co2_1000_msd_fit.png`
- `msd_co2_1000_result.json`
- `msd_co2_1000_result.csv`

## Tests

```bash
pytest
```

The current suite has 21 tests covering the regression, diffusivity calculation,
uncertainty helpers, model ranking, and the main CLI export switches.

## Physics

For the total 3-D MSD:

```text
D = (1 / 6) d(MSD_total) / dt
```

For each Cartesian direction:

```text
D_alpha = (1 / 2) d(MSD_alpha) / dt
```

The fitted slope has units of Angstrom squared per femtosecond. The conversion
used in the code is:

```text
1 Angstrom^2 / fs = 1e-5 m2/s
```
