# SimulationFiles

- All simulation files are examples. Each state point was run with 32 replicas: even simulation indices use the RxVB
  trial and odd indices the unbiased reaction trial.
- Use the default `--seed -1` so that each replica gets a different random seed.
- The launch scripts require FEASST and its Python package (`from feasst import fstio`).

## slit_pore

Square-well fluid in a slit pore with fixed reaction sites (FEASST 0.25.21).

- `launch_multisite_slit_pore.py` – writes and runs the FEASST input for every replica
- `a.txt`, `b.txt`, `c.txt`, `d.txt` – particle files (A/C fixed site, B/D fluid)
- `example_dens_n450/` – generated inputs for N_B = 450 in a 12σ box (Figs. 2 and 3, Table I):
  `…chemi000_fstin.txt` (RxVB), `…chemi001_fstin.txt` (unbiased), initial configuration, slab shape and parameters
- `occupancy_runs_2025_n450/` – inputs of the runs used for the slit-pore occupancies in Table III (FEASST 0.25.12,
  same model). The N_B = 50, 150 and 300 runs differ only in the number of particles and start from a site at
  z = 1.5451σ.

Each run fills the pore to N_B, then runs 10² equilibration cycles and 10⁴ production cycles of 10⁶ trials.
Options used for the paper (other options at their defaults):

| Sweep | Figure / table | Options |
|---|---|---|
| Density | Fig. 4 (middle), Tables I and III | `--tag dens --num_particles {50,150,300,450}` |
| System size | Fig. 4 (top) | `--tag syssize --density 0.78 --num_particles {50,150,300,450,600,800}` |
| Shell cutoff | Fig. 4 (bottom) | `--tag avdist --num_particles 450 --avb_outer {1.1,1.15,1.2,1.25,1.3,1.35}` |
| Multiple sites | Fig. 5, Table II | `--tag multi --placement spread --num_particles 450 --num_sites {1,2,4,6,9}` |

`--avb_outer` sets both the reaction-volume outer radius and the square-well cutoff.

Short test (8 replicas, about 1 minute):

```bash
cd SimulationFiles/slit_pore
python launch_multisite_slit_pore.py --tag dens --num_particles 50 --tpc 100000 --equilibration 2 --production 100 --weight 0.01 --procs_per_job 8
python ../../Scripts/analysis/racc_at_prod_end.py .
```

## mfi

Water in one MFI unit cell with one Brønsted acid site, T = 373 K.

### current_feasst

Launch script and particle files for FEASST 0.25.21, from the FEASST rxvb tutorial. Same model as the runs in the
paper; the particle files differ only in format. Short test (2 replicas, about 10 s):

```bash
cd SimulationFiles/mfi/current_feasst
python launch_2_acidic_zeolite.py --num_particles 50 --mu -550 --tpc 2000 --equilibration 5 --production 5 --procs_per_job 2
```

For the paper: `--tpc 1000000 --equilibration 10000 --procs_per_job 32`, `--num_particles` 1, 2, 3, 4, 6, 8, 10
and 50, and μ from −700 to −300 kJ/mol. Only the first stage (`*_eq*` files) was analysed; the replicas reached
about 2×10⁹ trials. In this version both reaction trials are attempted at the same rate.

### as_run_feasst_0.25.16

The files used for the MFI simulations in the paper (FEASST 0.25.16, `TrialRXNAVB`). They do not run with
FEASST 0.25.21.

- `launch_2_acidic_zeolite.py` – writes and runs the FEASST input for every replica
- `*.fstprt` – particle files: framework, acid site (`oh`), deprotonated site (`o-`), SPC/E water
  (`spce_with_dummy`), hydronium (`h3o`) and blocking spheres (`block`, with `MFI_Zeolite_111_1-54_blocks.txt`)
- `50w.slurm` – example Slurm array script for one loading
- `example_50w_mu550/` – generated inputs for 50 waters at μ = −550 kJ/mol (Fig. 8): `acid_zeo0_run.txt` (RxVB),
  `acid_zeo1_run.txt` (unbiased) and `acid_zeo_params.json`

File paths in the example inputs were changed to relative paths.
