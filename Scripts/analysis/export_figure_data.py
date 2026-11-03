#!/usr/bin/env python3
"""Write the FigureData/ CSVs from the raw FEASST output.

The raw output (per-replica logs, CPU-time files and trajectories, ~30 GB) is
not in this repository. This script records how each FigureData file was made.

Usage:
  python export_figure_data.py SLIT_DIR MFI_RUNS OUT_DIR

  SLIT_DIR  slit-pore production runs (dens_*, syssize_*, avdist_*, multi_spread_* prefixes)
  MFI_RUNS  MFI runs, laid out as {loading}w/mu_{|mu|}/acid_zeon0s{sim}_eq*.csv
  OUT_DIR   FigureData/

Run from a folder that holds the outputs of the other analysis scripts:
  mfi_distances.npz              mfi_distances.py MFI_RUNS/50w/mu_550           (Fig. 8)
  av_occupancy.csv               mfi_occupancy.py MFI_RUNS                      (Table III, MFI occupancy)
  slit_occupancy_production.txt  slit_occupancy.py RUN_DIR .xyz 10, per N_B     (Table III, slit occupancy)
"""
import csv
import glob
import os
import re
import sys

import numpy as np

import analyze_sweeps as A
from fig9_product_fraction import late_and_middle
from rebuild_efficiency_master import accepted_series

SLIT, MFI, OUT = sys.argv[1:4]
N450 = os.path.join(SLIT, "dens_flat_k1_n450_a1.05_chemi")
PROD_ROWS = 10000          # production = 1e4 cycles of 1e6 trials, one row per cycle
SCHEME = lambda s: "RxVB" if s % 2 == 0 else "unbiased"   # even sims biased, odd unbiased
# MFI state point used for Table III at each loading (near the Fig. 9 midpoint)
MFI_MIDPOINT = {"1w": -480, "2w": -535, "3w": -570, "4w": -580,
                "6w": -600, "8w": -590, "10w": -600, "50w": -585}


def write(name, header, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print("wrote", name, len(rows), "rows")


def pfx(name):
    return os.path.join(SLIT, name)


def mean_counts(prefix):
    """Mean accepted reactions per replica at the end of production: (RxVB, unbiased)."""
    P = A.prod_cycles(prefix)
    b = [A.num_accepted_at_production_end(f"{prefix}n0s{s}.csv", P) for s in range(0, 32, 2)]
    m = [A.num_accepted_at_production_end(f"{prefix}n0s{s}.csv", P) for s in range(1, 32, 2)]
    return np.mean(b), np.mean(m)


# Fig. 2: running average <N_C> per replica, cut at the end of production
cols = []
for s in range(32):
    with open(f"{N450}n0s{s}_num2.csv") as f:
        cols.append([float(r["average"]) for r in csv.DictReader(f)][:PROD_ROWS])
write("fig2_nc_running_average.csv",
      ["mc_trials"] + [f"sim{s}_{SCHEME(s)}" for s in range(32)],
      [[(i + 1) * 1_000_000] + [f"{c[i]:.6g}" for c in cols] for i in range(PROD_ROWS)])

# Fig. 3: block standard deviation of <N_C> vs CPU time, and the -1/2 slope intercepts
rows, fits = [], []
for s in range(32):
    t, sd = A.block_series(f"{N450}n0s{s}_cpu.csv", f"{N450}n0s{s}_num2.csv")
    rows += [[s, SCHEME(s), f"{a:.6g}", f"{b:.6g}"] for a, b in zip(t, sd)]
    fits.append([s, SCHEME(s), f"{A.intercept(N450, s):.6f}"])
write("fig3_block_stdev_vs_cpu.csv", ["sim", "scheme", "cpu_hours", "block_stdev_NC"], rows)
write("fig3_fit_intercepts.csv", ["sim", "scheme", "intercept_b"], fits)

# Figs. 4 and 5: R_acc and z12 sweeps (mean and SEM over 16 replica pairs)
def sweep(out, header, cfgs):
    rows = []
    for x, name, K in cfgs:
        r = A.analyze(pfx(name), K)
        rows.append(x + [f"{r['R_acc']:.6g}", f"{r['R_err']:.6g}", f"{r['z12']:.6g}", f"{r['z_err']:.6g}"])
    write(out, header + ["R_acc", "R_acc_sem", "z12", "z12_sem"], rows)


sweep("fig4a_system_size.csv", ["N_B"],
      [([n], f"syssize_flat_k1_n{n}_a1.05_chemi", 1) for n in (50, 150, 300, 450, 600, 800)])
sweep("fig4b_density.csv", ["N_B", "rho_sigma3"],   # rho*sigma^3 = N_B / accessible slab volume (4 x 12^2)
      [([n, f"{n / (4 * 12**2):.2g}"], f"dens_flat_k1_n{n}_a1.05_chemi", 1) for n in (50, 150, 300, 450)])
sweep("fig4c_shell_cutoff.csv", ["r_u_RxV_over_sigma"],
      [(["1.05"], "dens_flat_k1_n450_a1.05_chemi", 1)] +
      [([a], f"avdist_flat_k1_n450_a{a}_chemi", 1) for a in ("1.1", "1.15", "1.2", "1.25", "1.3", "1.35")])
sweep("fig5_multisite.csv", ["K"],
      [([k], f"multi_spread_k{k}_n450_a1.05_chemi", k) for k in (1, 2, 4, 6, 9)])

# Fig. 8: per-frame distances from O- in reacted frames (50w, mu = -550 kJ/mol)
z = np.load("mfi_distances.npz")
write("fig8_mfi_distances.csv", ["species", "r_angstrom"],
      [["H3O+_O", f"{v:.4f}"] for v in z["r_h3o"]] +
      [["nearest_water_O", f"{v:.4f}"] for v in z["r_pnear"]])

# Fig. 9: per-replica hydronium product fraction (final and middle thirds of the run)
rows = []
for lab in sorted(os.listdir(MFI), key=lambda p: int(re.sub(r"\D", "", p) or 0)):
    if not re.fullmatch(r"\d+w", lab):
        continue
    for md in sorted(d for d in os.listdir(os.path.join(MFI, lab)) if d.startswith("mu_")):
        for fn in sorted(os.listdir(os.path.join(MFI, lab, md))):
            m = re.fullmatch(r"acid_zeon0s(\d+)_eq_num3\.csv", fn)
            r = late_and_middle(os.path.join(MFI, lab, md, fn)) if m else None
            if r is not None:
                s = int(m.group(1))
                rows.append([lab, -float(md[3:]), s, SCHEME(s), f"{r[0]:.6g}", f"{r[1]:.6g}"])
write("fig9_product_fraction_per_replica.csv",
      ["loading", "mu_kJ_per_mol", "sim", "scheme", "theta_late_third", "theta_middle_third"], rows)

# Table I: slit pore, N_B = 450
r = A.analyze(N450, 1)
write("table1_slit_efficiency.csv", ["metric", "mean", "sem"],
      [["R_acc", f"{r['R_acc']:.2f}", f"{r['R_err']:.2f}"], ["z12", f"{r['z12']:.1f}", f"{r['z_err']:.1f}"]])

# Table II: accepted reactions per replica vs number of sites K
rows = []
for k in (1, 2, 4, 6, 9):
    p = pfx(f"multi_spread_k{k}_n450_a1.05_chemi")
    b, m = mean_counts(p)
    r = A.analyze(p, k)
    rows.append([k, f"{b:.1f}", f"{m:.1f}", f"{r['R_acc']:.2f}", f"{r['R_err']:.2f}"])
write("table2_multisite_counts.csv", ["K", "Nacc_biased", "Nacc_unbiased", "R_acc", "R_acc_sem"], rows)

# Table III: occupancy, accepted reactions, R_acc and z_max = N_B / <N_B^RxV | N_B^RxV >= 1>
occ = {}
for line in open("slit_occupancy_production.txt"):
    m = re.search(r"(?:dens|/)(\d+) \.xyz all : <n\|n>=1> = ([\d.]+) \+- ([\d.]+)", line)
    if m:
        occ[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
rows = []
for n in (50, 150, 300, 450):
    p = pfx(f"dens_flat_k1_n{n}_a1.05_chemi")
    b, m = mean_counts(p)
    r = A.analyze(p, 1)
    o, o_sem = occ[n]
    rows.append(["slit pore", n, f"{o:.4f}", f"{o_sem:.4f}", f"{b:.1f}", f"{m:.1f}",
                 f"{r['R_acc']:.2f}", f"{r['R_err']:.2f}", f"{n / o:.2f}"])
mocc = {r["label"]: r for r in csv.DictReader(open("av_occupancy.csv")) if r["system"].startswith("MFI")}
for lab, mu in MFI_MIDPOINT.items():
    # accepted reactions per replica over the whole run
    cnt = {0: [], 1: []}
    for fn in sorted(glob.glob(os.path.join(MFI, lab, f"mu_{abs(mu)}", "acid_zeon0s*_eq.csv"))):
        v = accepted_series(fn)
        if v is not None:
            cnt[int(re.search(r"0s(\d+)_eq\.csv$", fn).group(1)) % 2].append(v[-1])
    # the RxVB trial was attempted half as often as the unbiased trial: factor of two
    b, u = 2 * np.array(cnt[0]), np.array(cnt[1])
    R = b.mean() / u.mean()
    # SEM propagated from the SEM of each mean
    R_sem = R * np.hypot(b.std(ddof=1) / np.sqrt(b.size) / b.mean(), u.std(ddof=1) / np.sqrt(u.size) / u.mean())
    o = mocc[lab]
    n, oc = int(lab[:-1]), float(o["N_AV_given_occupied"])
    rows.append(["MFI", n, f"{oc:.4f}", f"{float(o['N_AV_given_occupied_sem']):.4f}",
                 f"{b.mean():.1f}", f"{u.mean():.1f}", f"{R:.2f}", f"{R_sem:.2f}", f"{n / oc:.2f}"])
write("table3_occupancy_and_speedup.csv",
      ["system", "N_B", "occupancy_given_occupied", "occupancy_sem", "Nacc_biased", "Nacc_unbiased",
       "R_acc", "R_acc_sem", "z_max"], rows)
