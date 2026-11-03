#!/usr/bin/env python3
"""Reaction-volume occupancy in MFI (Table III, MFI rows).

Counts water oxygens (site type 2) within 2.4-3.7 A of the protonated acid-site
oxygen (site type 0) in every reactant-state frame of acid_zeon0s*_eq.xyz,
at one chemical potential per loading. Per replica it averages the count over
all frames and over frames with at least one water in the shell; the table
reports the mean over replicas and its SEM.

Usage: python mfi_occupancy.py MFI_RUNS   (writes av_occupancy.csv)
"""
import csv
import glob
import os
import sys

import numpy as np

MIDPOINT = {"1w": -480, "2w": -535, "3w": -570, "4w": -580,
            "6w": -600, "8w": -590, "10w": -600, "50w": -585}
R_IN, R_OUT = 2.4, 3.7


def frames(path):
    """Yield (site_types, coords, box) for every frame of a FEASST xyz movie."""
    lines = open(path).read().split("\n")
    n = int(lines[0])
    box = np.array([float(v) for v in lines[1].split()[1:4]])
    for k in range(len(lines) // (n + 2)):
        blk = lines[k * (n + 2) + 2: k * (n + 2) + 2 + n]
        if len(blk) < n or not blk[0].strip():
            continue
        p = [ln.split() for ln in blk]
        if any(len(x) < 4 for x in p):
            continue
        yield np.array([int(x[0]) for x in p]), np.array([[float(v) for v in x[1:4]] for x in p]), box


def replica(path):
    """(<N>, <N | N >= 1>, P(N >= 1)) over the reactant-state frames of one replica."""
    counts = []
    for st, xy, box in frames(path):
        a = xy[st == 0]
        if len(a) == 0:               # deprotonated: product state
            continue
        d = xy[st == 2] - a[0]
        d -= box * np.round(d / box)
        r = np.linalg.norm(d, axis=1)
        counts.append(int(((r >= R_IN) & (r <= R_OUT)).sum()))
    c = np.array(counts, float)
    occ = c[c >= 1]
    return c.mean(), occ.mean() if len(occ) else np.nan, len(occ) / len(c)


if __name__ == "__main__":
    runs = sys.argv[1]
    sem = lambda v: v.std(ddof=1) / np.sqrt(len(v))
    with open("av_occupancy.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["system", "label", "n_b", "n_replicas", "N_AV", "N_AV_sem", "P_occupied",
                    "N_AV_given_occupied", "N_AV_given_occupied_sem"])
        for lab, mu in MIDPOINT.items():
            files = sorted(glob.glob(os.path.join(runs, lab, f"mu_{abs(mu)}", "acid_zeon0s*_eq.xyz")))
            s = np.array([replica(fn) for fn in files])
            s = s[np.isfinite(s).all(axis=1)]
            w.writerow(["MFI-111", lab, int(lab[:-1]), len(s), f"{s[:, 0].mean():.4f}", f"{sem(s[:, 0]):.4f}",
                        f"{s[:, 2].mean():.4f}", f"{s[:, 1].mean():.4f}", f"{sem(s[:, 1]):.4f}"])
            print(f"{lab:>4}: <N | N >= 1> = {s[:, 1].mean():.4f} ± {sem(s[:, 1]):.4f} ({len(s)} replicas)")
    print("wrote av_occupancy.csv")
