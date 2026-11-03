# FigureData

Even replica indices (`sim0`, `sim2`, ...) use the RxVB trial and odd indices the unbiased reaction trial.
Means and standard errors (SEM) are over the 16 replicas of each scheme.

| File | Figure / table | Contents |
|---|---|---|
| `fig2_nc_running_average.csv` | Fig. 2 | Running average of the number of product C particles vs MC trials, 32 replicas (slit pore, N_B = 450) |
| `fig3_block_stdev_vs_cpu.csv` | Fig. 3 | Block standard deviation of ⟨N_C⟩ vs CPU hours, per replica |
| `fig3_fit_intercepts.csv` | Fig. 3, Table I | Intercept b of ln σ = −½ ln t + b per replica; z12 = exp[2(b_unbiased − b_RxVB)] |
| `fig4a_system_size.csv` | Fig. 4 (top) | R_acc and z12 vs N_B at ρσ³ = 0.78 |
| `fig4b_density.csv` | Fig. 4 (middle) | R_acc and z12 vs N_B and ρσ³ in a 12σ box |
| `fig4c_shell_cutoff.csv` | Fig. 4 (bottom) | R_acc and z12 vs reaction-volume cutoff r_u^RxV/σ, N_B = 450 |
| `fig5_multisite.csv` | Fig. 5 | R_acc and z12 vs number of reaction sites K, N_B = 450 |
| `fig8_mfi_distances.csv` | Fig. 8 | Distance from O⁻ to the hydronium O and to the nearest water O in each reacted frame (MFI, 50 waters, μ = −550 kJ/mol) |
| `fig9_product_fraction_per_replica.csv` | Fig. 9 | Hydronium product fraction per replica, loading and μ, over the final and middle thirds of each run |
| `fig9_mu_star.csv` | Fig. 9 | Midpoint μ* per loading and scheme with its 95 % bootstrap interval |
| `table1_slit_efficiency.csv` | Table I | R_acc and z12, slit pore, N_B = 450 |
| `table2_multisite_counts.csv` | Table II | Accepted reactions per replica and R_acc vs K |
| `table3_occupancy_and_speedup.csv` | Table III, Fig. 10 | Reaction-volume occupancy, accepted reactions, R_acc and z_max |
| `slit_racc_at_production_end.csv` | Tables I–III, Figs. 4–5 | Slit-pore R_acc for each run, counted at the end of production (`racc_new`) and at the last log row (`racc_old`) |
| `slit_occupancy_production.txt` | Table III | Slit-pore reaction-volume occupancy |
| `mfi_efficiency_master.csv` | Table III | Accepted reactions and R_acc for every MFI loading and μ |

## Notes

- Accepted reactions are counted at the end of the production stage. Replicas that finish early continue to sample
  until all replicas have finished; these cycles are not counted.
- z12: the slope of ln σ vs ln t is fixed at −½, and the fit starts 5×10⁶ trials into production.
- R_acc uncertainties are propagated from the SEM of each mean.
- In the MFI runs the RxVB trial was attempted half as often as the unbiased trial, so the MFI N_acc and R_acc in
  Table III include a factor of two.
- MFI rows of Table III use one μ per loading: −480 (1w), −535 (2w), −570 (3w), −580 (4w), −600 (6w), −590 (8w),
  −600 (10w) and −585 kJ/mol (50w). MFI occupancies are averaged over all reactant-state frames.
- The slit-pore occupancies are from the production trajectories of the runs in
  `SimulationFiles/slit_pore/occupancy_runs_2025_n450/`, which use the same model.
- In the shell-cutoff runs (Fig. 4, bottom) the square-well range equals r_u^RxV. In the multi-site runs a single
  site is at the pore centre.
- `mfi_efficiency_master.csv`: `n_rows` is the median number of logged rows per replica (10⁶ trials each);
  `Nacc_biased` and `Nacc_unbiased` are mean accepted reactions per replica over the whole run (`*_med` medians,
  `*_zeros` replicas with none); `R_acc_total` is the ratio of the means with a bootstrap SEM, without the factor
  of two; `R_acc_late` is the same over the final third of the run.
- Fig. 9 is saved as `fig6_biased.png` and `fig6_unbiased.png`.
