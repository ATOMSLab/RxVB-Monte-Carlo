#!/usr/bin/env python3
"""Redraw Figs. 2, 3, 4, 5, 8, 9 and 10 of the paper from FigureData/.

Usage (from the repository root):  python Scripts/figures/plot_figures.py
Writes PNGs to figures_out/ (file names as in the manuscript). Needs numpy and
matplotlib; Fig. 9 also needs scipy and pandas (it reuses Scripts/analysis/fig9_product_fraction.py).
"""
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DATA = "FigureData"
OUT = "figures_out"
RED = "#B74B3E"    # unbiased / R_acc / H3O+ / MFI
BLUE = "#3E6FB7"   # RxVB / z12 / water / slit pore
CREAM = "#f6f4ea"
os.makedirs(OUT, exist_ok=True)

plt.rc("font", family="sans-serif", size=16)
plt.rcParams["mathtext.default"] = "regular"
plt.rcParams["mathtext.fontset"] = "dejavusans"


def read(name):
    with open(os.path.join(DATA, name)) as f:
        return list(csv.DictReader(f))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=300, facecolor="white")
    plt.close(fig)
    print("wrote", os.path.join(OUT, name))


# ---- Fig. 2: <N_C> vs MC trials ----
rows = read("fig2_nc_running_average.csv")
x = np.array([float(r["mc_trials"]) for r in rows])
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_facecolor(CREAM)
sims = [k for k in rows[0] if k.startswith("sim")]
for k in sorted(sims, key=lambda k: "RxVB" in k):   # unbiased first, RxVB drawn on top
    ax.plot(x, [float(r[k]) for r in rows], color=BLUE if "RxVB" in k else RED, lw=0.8, alpha=0.9)
ax.set_xscale("log"); ax.set_xlim(1e6, 1e10); ax.set_ylim(0.0, 1.02)
ax.set_xlabel("MC trials", fontsize=22); ax.set_ylabel(r"$\langle N_C\rangle$", fontsize=22)
ax.tick_params(labelsize=18)
ax.legend([Line2D([0], [0], color=RED, lw=2), Line2D([0], [0], color=BLUE, lw=2)],
          ["unbiased", "RxVB"], fontsize=16, frameon=False)
save(fig, "nc_av.png")

# ---- Fig. 3: ln sigma vs ln t with -1/2 slope fits ----
rows = read("fig3_block_stdev_vs_cpu.csv")
fits = {int(r["sim"]): float(r["intercept_b"]) for r in read("fig3_fit_intercepts.csv")}
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_facecolor(CREAM)
for s in range(32):
    t = np.array([float(r["cpu_hours"]) for r in rows if int(r["sim"]) == s])
    sd = np.array([float(r["block_stdev_NC"]) for r in rows if int(r["sim"]) == s])
    m = (t > 0) & (sd > 0)
    lnt, lns = np.log(t[m]), np.log(sd[m])
    keep = (lnt >= -3.0) & (lns >= -9)
    ax.scatter(lnt[keep], lns[keep], s=4, color=BLUE if s % 2 == 0 else RED, alpha=0.25, edgecolors="none")
    xf = np.linspace(max(np.log(t[5]), -3.0), lnt.max(), 30)   # fit range: from 5e6 trials (row 5) on
    ax.plot(xf, -0.5 * xf + fits[s], color="black", lw=1.1, alpha=0.6)
b_rxvb = np.mean([b for s, b in fits.items() if s % 2 == 0])
b_unb = np.mean([b for s, b in fits.items() if s % 2 == 1])
print(f"z12 = exp[2(b_unbiased - b_RxVB)] = {np.exp(2 * (b_unb - b_rxvb)):.1f}")
ax.set_xlim(-3, 3); ax.set_ylim(-8, -2)
ax.set_xlabel(r"$\ln t$ (CPU-h)", fontsize=22); ax.set_ylabel(r"$\ln \sigma_{\langle N_C\rangle}$", fontsize=22)
ax.tick_params(labelsize=18)
ax.legend([Line2D([0], [0], marker="o", color=RED, lw=0, ms=7), Line2D([0], [0], marker="o", color=BLUE, lw=0, ms=7)],
          ["unbiased", "RxVB"], fontsize=16, frameon=False, loc="upper right")
save(fig, "lnsig_nc_av.png")


# ---- Figs. 4 and 5: R_acc and z12 bar charts ----
def bars(src, xcol, labels, xlabel, bar_width, caps, out):
    rows = read(src)
    xs = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for i, (key, lab, c) in enumerate([("R_acc", r"$\mathrm{R}_\mathrm{acc}$", RED), ("z12", r"$\mathrm{z}_{12}$", BLUE)]):
        kw = dict(error_kw=dict(ecolor="black", capsize=3, elinewidth=1)) if caps else {}
        ax.bar(xs + i * bar_width, [float(r[key]) for r in rows], yerr=[float(r[key + "_sem"]) for r in rows],
               width=bar_width, label=lab, color=c, alpha=0.85, edgecolor="black", zorder=2, **kw)
    ax.set_facecolor(CREAM)
    ax.grid(axis="y", color="white", linewidth=1, zorder=1)
    ax.grid(axis="y", linestyle=":", linewidth=0.5)
    ax.set_xticks(xs + bar_width / 2); ax.set_xticklabels(labels(rows, xcol))
    ax.set_ylabel("Speedup (×)", fontsize=22); ax.set_xlabel(xlabel, fontsize=22)
    ax.tick_params(axis="both", labelsize=18)
    ax.legend(frameon=False); ax.margins(x=0.03)
    save(fig, out)


col = lambda rows, k: [r[k] for r in rows]
bars("fig4a_system_size.csv", "N_B", col, r"$N_B$", 0.35, True, "syssize_effect.png")
bars("fig4b_density.csv", "rho_sigma3", col, r"$\rho\sigma^3$", 0.25, False, "dens_effect.png")
bars("fig4c_shell_cutoff.csv", "r_u_RxV_over_sigma", col, r"$r^{\mathrm{RxV}}_{u}/\sigma$", 0.3, False, "avb_effect.png")
bars("fig5_multisite.csv", "K", col, r"Number of acid sites, $K$", 0.35, True, "multisite_effect.png")

# ---- Fig. 8: distance distributions from O- (Gaussian KDE, bandwidth 0.04 A) ----
rows = read("fig8_mfi_distances.csv")
rmax, bw = 13.383 / 2, 0.04          # half the shortest MFI box length
grid = np.linspace(0, rmax, 700)


def kde(v):
    out = np.zeros_like(grid)
    for chunk in np.array_split(v, max(1, v.size // 5000)):
        out += np.exp(-0.5 * ((grid[:, None] - chunk[None, :]) / bw) ** 2).sum(axis=1)
    return out / (v.size * bw * np.sqrt(2 * np.pi))


plt.rc("font", family="sans-serif", size=14)
fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
ax.set_facecolor(CREAM)
ax.grid(axis="y", color="white", linewidth=1, zorder=0)
for sp, c, lab, xy, ha in [("H3O+_O", RED, r"$\mathrm{H_3O^+}$", (2.55, 2.15), "right"),
                           ("nearest_water_O", BLUE, r"nearest $\mathrm{H_2O}$", (3.04, 2.25), "left")]:
    y = kde(np.array([float(r["r_angstrom"]) for r in rows if r["species"] == sp]))
    ax.fill_between(grid, y, color=c, alpha=0.2, lw=0, zorder=2)
    ax.plot(grid, y, color=c, lw=1.8, zorder=5)
    ax.text(*xy, lab, color=c, fontsize=14, ha=ha, zorder=4, bbox=dict(facecolor=CREAM, edgecolor="none", pad=1.5))
for r in (2.4, 3.7):                 # RxV shell
    ax.axvline(r, color="0.35", ls="--", lw=1, zorder=1)
ax.set_ylim(0, 2.8); ax.set_xlim(0, rmax)
ax.text(3.05, 2.72, "RxV shell", ha="center", va="top", color="0.35", fontsize=12)
ax.set_xlabel(r"$r$ from $\mathrm{O^-}$ (Å)"); ax.set_ylabel(r"Probability density (Å$^{-1}$)")
save(fig, "mfi_dist_5_7.png")

# ---- Fig. 10: R_acc vs z_max parity ----
rows = read("table3_occupancy_and_speedup.csv")
plt.rc("font", family="sans-serif", size=18)
fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=150)
ax.set_facecolor(CREAM)
lim = [0.4, 700]
ax.plot(lim, lim, ls="--", color="0.45", lw=1.3, zorder=1)
offsets = {("slit pore", "50"): (12, -4), ("slit pore", "150"): (-14, -10), ("slit pore", "300"): (-3, -14),
           ("slit pore", "450"): (-14, 2), ("MFI", "1"): (9, -5), ("MFI", "2"): (8, -5), ("MFI", "3"): (-20, 7),
           ("MFI", "4"): (2, -14), ("MFI", "6"): (-2, 9), ("MFI", "8"): (8, -4), ("MFI", "10"): (7, 6), ("MFI", "50"): (9, -2)}
for r in rows:
    slit = r["system"] == "slit pore"
    zx, ry = float(r["z_max"]), float(r["R_acc"])
    dx, dy = offsets[(r["system"], r["N_B"])]
    ax.scatter(zx, ry, marker="o" if slit else "s", s=70 if slit else 48, color=BLUE if slit else RED,
               edgecolors="none", zorder=3)
    lab = rf"$N_B={r['N_B']}$" if slit else f"{r['N_B']}w"
    ha = "left" if (not slit or dx >= 0 or r["N_B"] == "300") else "right"
    ax.annotate(lab, (zx, ry), textcoords="offset points", xytext=(dx, dy), fontsize=15, color="0.35",
                va="center" if slit else "baseline", ha=ha)
ax.legend([Line2D([0], [0], marker="o", color=BLUE, lw=0, ms=11), Line2D([0], [0], marker="s", color=RED, lw=0, ms=10)],
          ["Slit pore model", r"Br$\o$nsted acid MFI"], fontsize=16, frameon=False, loc="upper left")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_aspect("equal", adjustable="box")
ax.grid(True, which="major", color="white", linewidth=1.3, zorder=0)
ax.grid(True, which="minor", color="white", linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
ax.set_xlabel(r"Maximum speedup, $z_\mathrm{max}$", fontsize=26)
ax.set_ylabel(r"Observed speedup, $R_\mathrm{acc}$", fontsize=26)
ax.tick_params(axis="both", labelsize=22)
save(fig, "av_parity.png")

# ---- Fig. 9: product fraction vs mu, sigmoid fits with bootstrap bands ----
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis"))
import fig9_product_fraction as f9   # noqa: E402

plt.rcdefaults()                     # Fig. 9 was drawn with matplotlib's default style
per = {"biased": {}, "unbiased": {}}
for r in read("fig9_product_fraction_per_replica.csv"):
    sch = "biased" if r["scheme"] == "RxVB" else "unbiased"
    per[sch].setdefault(r["loading"], {}).setdefault(float(r["mu_kJ_per_mol"]), []).append(
        (float(r["theta_late_third"]), float(r["theta_middle_third"])))
for sch in ("biased", "unbiased"):
    data = {}
    for lab in sorted(per[sch], key=lambda x: int(x[:-1])):
        data[lab] = {}
        for mu, v in per[sch][lab].items():
            if len(v) >= 4:
                late = np.array([a for a, _ in v]); drift = late - np.array([b for _, b in v])
                data[lab][mu] = (late, drift.mean(), drift.std(ddof=1) / np.sqrt(len(drift)))
    f9.make_figure(f9.analyse(data), os.path.join(OUT, f"fig6_{sch}.png"))
