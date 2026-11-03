#!/usr/bin/env python3
"""Accepted reactions and R_acc for every MFI state point.

num_accepted is a cumulative counter, so per replica
    N_acc (whole run)   = v[-1]
    N_acc (final third) = v[-1] - v[2n/3]
where v is num_accepted summed over the reaction-trial columns. R_acc is the
ratio of the RxVB and unbiased means (no factor of two applied) with a
bootstrap SEM over replicas.

Run from the folder that contains MFI-111_1p/runs/temp373/.
Writes mfi_efficiency_master.csv.
"""
import glob
import os
import re

import numpy as np
import pandas as pd

RUNS = "MFI-111_1p/runs/temp373"
RNG = np.random.default_rng(0)


def accepted_series(path):
    """Cumulative accepted-reaction count per row, summed over reaction trials."""
    try:
        hdr = pd.read_csv(path, nrows=0)
    except Exception:
        return None
    cols = [c for c in hdr.columns if c.strip().startswith("num_accepted")]
    if not cols:
        return None
    try:
        df = pd.read_csv(path, usecols=cols, low_memory=False)
    except Exception:
        return None
    if len(df) < 6:
        return None
    v = df.apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(float).sum(axis=1)
    # cumulative counters must not decrease; a checkpoint rollback shows up as a
    # drop, so rebuild the surviving chain the same way as for moment0
    keep = np.zeros(len(v), bool)
    lim = np.inf
    for i in range(len(v) - 1, -1, -1):
        if v[i] < lim + 1e-9:
            keep[i] = True
            lim = v[i]
    return v[keep]


def state_point(mu_dir):
    tot = {0: [], 1: []}
    seg = {0: [], 1: []}
    nrow = []
    for f in sorted(glob.glob(os.path.join(mu_dir, "acid_zeon0s*_eq.csv"))):
        m = re.search(r"0s(\d+)_eq\.csv$", f)
        if not m:
            continue
        v = accepted_series(f)
        if v is None:
            continue
        n = len(v)
        one = n // 3
        tot[int(m.group(1)) % 2].append(v[-1])
        seg[int(m.group(1)) % 2].append(v[-1] - v[2 * one])
        nrow.append(n)
    if not tot[0] or not tot[1]:
        return None
    b, r = np.array(tot[0]), np.array(tot[1])
    sb, sr = np.array(seg[0]), np.array(seg[1])

    def ratio_boot(x, y):
        if y.mean() <= 0:
            return np.nan, np.nan
        boots = []
        for _ in range(2000):
            yy = RNG.choice(y, len(y), replace=True).mean()
            if yy > 0:
                boots.append(RNG.choice(x, len(x), replace=True).mean() / yy)
        return x.mean() / y.mean(), (np.std(boots) if boots else np.nan)

    R_tot, R_tot_sem = ratio_boot(b, r)
    R_seg, R_seg_sem = ratio_boot(sb, sr)
    return dict(
        n_rows=int(np.median(nrow)),
        Nacc_biased=b.mean(), Nacc_unbiased=r.mean(),
        Nacc_biased_med=float(np.median(b)), Nacc_unbiased_med=float(np.median(r)),
        Nacc_min=float(min(b.min(), r.min())),
        biased_zeros=int((b == 0).sum()), unbiased_zeros=int((r == 0).sum()),
        R_acc_total=R_tot, R_acc_total_sem=R_tot_sem,
        R_acc_late=R_seg, R_acc_late_sem=R_seg_sem,
    )


def main():
    rows = []
    for ld in sorted(glob.glob(os.path.join(RUNS, "*w")),
                     key=lambda p: int(re.sub(r"\D", "", os.path.basename(p)))):
        lab = os.path.basename(ld)
        for d in sorted(glob.glob(os.path.join(ld, "mu_*")),
                        key=lambda p: int(re.sub(r"\D", "", os.path.basename(p)))):
            s = state_point(d)
            if s is None:
                continue
            rows.append(dict(w=lab, mu=-int(re.sub(r"\D", "", os.path.basename(d))), **s))
        print(f"  {lab}: done")

    df = pd.DataFrame(rows)
    df["min_Nacc"] = df[["Nacc_biased_med", "Nacc_unbiased_med"]].min(axis=1)
    df.to_csv("mfi_efficiency_master.csv", index=False)
    print(f"\nwrote mfi_efficiency_master.csv  ({len(df)} state points)")


if __name__ == "__main__":
    main()
