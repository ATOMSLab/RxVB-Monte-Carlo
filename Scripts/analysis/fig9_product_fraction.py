#!/usr/bin/env python3
"""Fig. 9: hydronium product fraction vs reaction chemical potential in MFI, with
sigmoid fits and bootstrapped 95 % bands, one panel per trial scheme.

Per replica the product fraction is the mean of the de-cumulated block averages of
the H3O+ count (acid_zeon0s{sim}_eq_num3.csv) over the final third of the run; the
middle third is used for the convergence flag. Even sims = RxVB, odd = unbiased.

Usage:  python fig9_product_fraction.py MFI_RUNS
        (MFI_RUNS laid out as {loading}w/mu_{|mu|}/acid_zeon0s{sim}_eq_num3.csv)
Writes fig6_biased.png and fig6_unbiased.png (the manuscript file names) and fig9_mu_star.csv.
"""
import csv
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

OBS = "num3"                         # num3 = hydronium; num2 = O- (identical, coupled 1:1)

# convergence test: final third vs middle third
DRIFT_K = 2.0        # a drift counts if |drift| > DRIFT_K * SEM
DRIFT_FLOOR = 0.02   # and |drift| > DRIFT_FLOOR

# bootstrap for the fitted-curve band and the mu* interval
NBOOT = 1000
RNG = np.random.default_rng(7)

ORDER = ["1w", "2w", "3w", "4w", "6w", "8w", "10w", "50w"]
BG = "#f6f4ea"
PALETTE = {"1w": "#004c6d", "2w": "#2e9bd5", "3w": "#a750dd", "4w": "#228C05",
           "6w": "#eeb404", "8w": "#a9c604", "10w": "#c92f08", "50w": "#952202"}
GRID = np.linspace(-710, -430, 600)


def drop_rollbacks(m0):
    """Keep only the surviving chain after checkpoint restarts."""
    keep, limit = np.zeros(len(m0), bool), np.inf
    for i in range(len(m0) - 1, -1, -1):
        if m0[i] < limit:
            keep[i], limit = True, m0[i]
    return keep


def late_and_middle(path):
    """(late-third mean, middle-third mean) of the per-block product fraction."""
    try:
        d = pd.read_csv(path, usecols=["average", "moment0"])
    except Exception:
        return None
    if len(d) < 9:
        return None
    m0, avg = d["moment0"].to_numpy(float), d["average"].to_numpy(float)
    k = drop_rollbacks(m0)
    m0, avg = m0[k], avg[k]
    if len(m0) < 9:
        return None
    blk = np.concatenate([[avg[0]], np.diff(avg * m0) / np.diff(m0)])   # de-cumulate
    n = len(blk) // 3
    return blk[2 * n:].mean(), blk[n:2 * n].mean()


def load(runs, scheme):
    """{loading: {mu: (late_values, drift_mean, drift_sem)}} for one scheme."""
    data = {}
    for ld in sorted(glob.glob(os.path.join(runs, "*w")),
                     key=lambda p: int(re.sub(r"\D", "", os.path.basename(p)))):
        lab, per_mu = os.path.basename(ld), {}
        for md_ in sorted(glob.glob(os.path.join(ld, "mu_*"))):
            mu = -float(re.sub(r"\D", "", os.path.basename(md_)))
            late, drift = [], []
            for f in glob.glob(os.path.join(md_, f"acid_zeon0s*_eq_{OBS}.csv")):
                m = re.search(rf"acid_zeon0s(\d+)_eq_{OBS}\.csv$", f)
                if not m:
                    continue
                if ("biased" if int(m.group(1)) % 2 == 0 else "unbiased") != scheme:
                    continue
                r = late_and_middle(f)
                if r is None:
                    continue
                late.append(r[0]); drift.append(r[0] - r[1])
            if len(late) >= 4:
                late, drift = np.array(late), np.array(drift)
                per_mu[mu] = (late, drift.mean(), drift.std(ddof=1) / np.sqrt(len(drift)))
        if per_mu:
            data[lab] = per_mu
    return data


def sigmoid(mu, mu0, w):
    return 1.0 / (1.0 + np.exp(-(mu - mu0) / w))


def fit_midpoint(mus, fracs):
    """Fitted mu*, or nan if the transition is not captured by the mu grid."""
    if len(mus) < 3 or np.ptp(fracs) < 0.2:
        return np.nan
    try:
        p, _ = curve_fit(sigmoid, mus, fracs,
                         p0=[mus[np.argmin(np.abs(fracs - 0.5))], 10.0], maxfev=20000)
    except Exception:
        return np.nan
    return p[0] if mus.min() - 60 < p[0] < mus.max() + 60 else np.nan


def converged(dm, ds):
    """Per point: do the final and middle thirds agree?"""
    ok = (np.abs(dm) <= DRIFT_K * np.where(ds > 0, ds, np.inf)) | (np.abs(dm) < DRIFT_FLOOR)
    return ok | ((ds == 0) & (np.abs(dm) == 0))


def fit_converged(dm, ds, trans):
    """Per loading: is the mean drift over the transition region within error?"""
    if not trans.any():
        return True
    md_ = abs(dm[trans].mean())
    sem = np.sqrt((ds[trans] ** 2).sum()) / trans.sum()
    return bool(md_ <= DRIFT_K * sem or md_ < DRIFT_FLOOR)


def analyse(data):
    """Per loading: point stats, convergence flags, fitted curve, band, mu* CI."""
    out = {}
    for lab, per_mu in data.items():
        mus = np.array(sorted(per_mu))
        mean = np.array([per_mu[m][0].mean() for m in mus])
        sem = np.array([per_mu[m][0].std(ddof=1) / np.sqrt(len(per_mu[m][0])) for m in mus])
        dm = np.array([per_mu[m][1] for m in mus])
        ds = np.array([per_mu[m][2] for m in mus])

        trans = (mean > 0.02) & (mean < 0.98)
        rec = dict(mus=mus, mean=mean, sem=sem, conv=converged(dm, ds),
                   cfit=fit_converged(dm, ds, trans),
                   m0=fit_midpoint(mus, mean), p=None, band=None, ci=(np.nan, np.nan))

        if rec["m0"] == rec["m0"]:
            rec["p"], _ = curve_fit(sigmoid, mus, mean, p0=[rec["m0"], 8.0], maxfev=20000)
            curves, mstars = [], []
            for _ in range(NBOOT):
                bm = np.array([per_mu[m][0][RNG.integers(0, len(per_mu[m][0]),
                                            len(per_mu[m][0]))].mean() for m in mus])
                v = fit_midpoint(mus, bm)
                if v != v:
                    continue
                try:
                    q, _ = curve_fit(sigmoid, mus, bm, p0=[v, 8.0], maxfev=20000)
                except Exception:
                    continue
                curves.append(sigmoid(GRID, *q)); mstars.append(q[0])
            if len(curves) > 30:
                rec["band"] = np.percentile(np.array(curves), [2.5, 97.5], axis=0)
                rec["ci"] = tuple(np.percentile(mstars, [2.5, 97.5]))
        out[lab] = rec
    return out


def make_figure(res_one, outfile):
    """One stacked single-column panel (3.5 in wide)."""
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.set_facecolor(BG); ax.set_axisbelow(True)
    ax.grid(axis="y", color="white", lw=0.9)

    for lab in ORDER:
        r = res_one.get(lab)
        if r is None:
            continue
        c = PALETTE[lab]
        if r["band"] is not None:
            ax.fill_between(GRID, r["band"][0], r["band"][1], color=c, alpha=0.22, lw=0, zorder=2)
        if r["p"] is not None:
            ax.plot(GRID, sigmoid(GRID, *r["p"]), color=c, lw=1.1,
                    ls="-" if r["cfit"] else "--", zorder=3, label=lab)
        k = r["conv"]
        # Bounded 95 % interval via the logit transform (delta method): equals the plain
        # +/-1.96 SEM bar in the interior and can never leave [0, 1].
        m_, s_ = r["mean"], r["sem"]
        z, lo, hi = 1.96, np.zeros_like(m_), np.zeros_like(m_)
        ok = (s_ > 0) & (m_ > 0) & (m_ < 1)
        p = np.clip(m_[ok], 1e-6, 1 - 1e-6)
        se_logit = s_[ok] / (p * (1 - p))
        l0 = np.log(p / (1 - p))
        lower = 1 / (1 + np.exp(-(l0 - z * se_logit)))
        upper = 1 / (1 + np.exp(-(l0 + z * se_logit)))
        lo[ok] = np.clip(m_[ok] - lower, 0, None)
        hi[ok] = np.clip(upper - m_[ok], 0, None)
        yerr = np.vstack([lo, hi])
        ax.errorbar(r["mus"][k], m_[k], yerr=yerr[:, k], ls="none",
                    marker="o", ms=2, color=c, capsize=2, lw=0.9, zorder=5)
        ax.errorbar(r["mus"][~k], m_[~k], yerr=yerr[:, ~k], ls="none",
                    marker="o", ms=2, mfc="white", mec=c, ecolor=c, capsize=2,
                    lw=0.9, mew=1.1, zorder=5)

    ax.axhline(0.5, color="0.45", ls="--", lw=0.8, zorder=1)
    ax.set_xlim(-700, -440)
    ax.set_ylim(-0.05, 1.2)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel(r"Reaction chemical potential ($\Delta\mu^{*}_{\mathrm{rxn}}$, kJ/mol)", fontsize=9)
    ax.set_ylabel(r"Product fraction ($\theta_{\mathrm{H}_3\mathrm{O}^+}$)", fontsize=10)
    ax.tick_params(labelsize=8.5)
    ax.legend(loc="upper left", fontsize=7, frameon=False, ncol=1,
              labelspacing=0.25, handlelength=1.3, borderpad=0.2)
    fig.tight_layout(); fig.savefig(outfile, dpi=300, facecolor="white")
    plt.close(fig)
    print("wrote", outfile)


if __name__ == "__main__":
    raw = {s: load(sys.argv[1], s) for s in ("biased", "unbiased")}
    res = {s: analyse(raw[s]) for s in ("biased", "unbiased")}
    make_figure(res["biased"], "fig6_biased.png")
    make_figure(res["unbiased"], "fig6_unbiased.png")
    with open("fig9_mu_star.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scheme", "loading", "mu_star", "ci_low", "ci_high", "settled"])
        for scheme in ("biased", "unbiased"):
            for lab in ORDER:
                r = res[scheme].get(lab)
                if r is None or r["m0"] != r["m0"]:
                    continue
                w.writerow([scheme, lab, round(r["m0"], 1), round(r["ci"][0], 1),
                            round(r["ci"][1], 1), "yes" if r["cfit"] else "no"])
    print("wrote fig9_mu_star.csv")
