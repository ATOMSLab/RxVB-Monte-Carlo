# RxVB_Chemisorption

Supporting information for "A reaction volume bias Monte Carlo trial for sampling chemisorption in confinement"
(S. Sharlin, H. W. Hatch, D. W. Siderius, T. R. Josephson).

The RxVB trial is implemented in [FEASST](https://github.com/usnistgov/feasst) (`plugin/rxvb`).

- `FigureData/` – data for each figure and table (see `FigureData/README.md`)
- `SimulationFiles/` – FEASST inputs for the slit-pore and MFI simulations (see `SimulationFiles/README.md`)
- `Scripts/analysis/` – analysis of the raw simulation output
- `Scripts/figures/plot_figures.py` – redraws Figs. 2–5 and 8–10 from `FigureData/`

## Figures

```bash
python Scripts/figures/plot_figures.py
```

Writes the figures to `figures_out/`. Requires numpy, scipy, pandas and matplotlib.

## Analysis

The raw simulation output (about 30 GB) is not included. From the raw output, `FigureData/` is produced with:

| Script | Input | Output |
|---|---|---|
| `racc_at_prod_end.py` | slit-pore runs | R_acc for each run (`racc_prodend.csv`) |
| `slit_occupancy.py` | slit-pore trajectories | reaction-volume occupancy |
| `rebuild_efficiency_master.py` | MFI runs | accepted reactions and R_acc for each state point |
| `mfi_occupancy.py` | MFI trajectories | reaction-volume occupancy (`av_occupancy.csv`) |
| `mfi_distances.py` | MFI trajectories, 50 waters, μ = −550 kJ/mol | distances for Fig. 8 |
| `fig9_product_fraction.py` | MFI runs | Fig. 9 and μ* |
| `export_figure_data.py` | all of the above | `FigureData/*.csv` |

`analyze_sweeps.py` contains the R_acc and z12 functions used by `export_figure_data.py`.

## Software

- Slit-pore simulations: FEASST 0.25.21. The occupancy runs in `SimulationFiles/slit_pore/occupancy_runs_2025_n450/`
  used FEASST 0.25.12.
- MFI simulations: FEASST 0.25.16, in which the RxVB trial was named `TrialRXNAVB`.
- Analysis: Python 3 with numpy, scipy, pandas and matplotlib.
