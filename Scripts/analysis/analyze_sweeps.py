#!/usr/bin/env python3
"""R_acc and z12 for the slit-pore runs.

For one run prefix with 32 replicas (even sims RxVB, odd sims unbiased):
  R_acc = <N_acc RxVB> / <N_acc unbiased>, with N_acc the summed num_accepted
          columns of the log at the end of production; SEM propagated from the
          SEM of each mean.
  z12   = exp[2(<b_unbiased> - <b_RxVB>)], where b is the intercept of
          ln(block stdev of <N_C>) = -1/2 ln(t_cpu) + b (slope fixed), fitted
          from 5e6 trials into production to the end of the log.

Each input ends with WriteFileAndCheck and Run until_file_exists, so a replica
that finishes early keeps running until all replicas are done. N_acc is read at
the last production row to exclude those cycles.

Usage: python analyze_sweeps.py RESULTS_DIR   (prints R_acc and z12 for each sweep)
"""
import csv
import glob
import math
import os
import re
import sys

import numpy as np

EQUIL_TRIALS = 5e6      # z12 fit starts this many trials into production
TRIALS_PER_ROW = 1e6    # one row of the num2/cpu files per cycle of 1e6 trials


def last_value(fn, name):
    """Last numeric value of column `name` in a csv file."""
    if not os.path.exists(fn):
        return None
    rows = list(csv.reader(open(fn)))
    if not rows or name not in rows[0]:
        return None
    i = rows[0].index(name)
    for r in reversed(rows[1:]):
        try:
            return float(r[i])
        except (ValueError, IndexError):
            pass
    return None


def block_series(cpu_fn, num2_fn):
    """CPU hours and block standard deviation of <N_C>, row by row."""
    t = []
    if os.path.exists(cpu_fn):
        for line in open(cpu_fn):
            p = line.split()
            if len(p) >= 2:
                try:
                    t.append(float(p[1]))
                except ValueError:
                    pass
    s = []
    if os.path.exists(num2_fn):
        for r in csv.reader(open(num2_fn)):
            if len(r) > 2:
                try:
                    s.append(float(r[2]))
                except ValueError:
                    pass
    n = min(len(t), len(s))
    return t[:n], s[:n]


def intercept(prefix, sim, equil_trials=EQUIL_TRIALS):
    """Intercept b of ln(sigma) = -1/2 ln(t) + b for one replica."""
    t, s = block_series(f"{prefix}n0s{sim}_cpu.csv", f"{prefix}n0s{sim}_num2.csv")
    if len(t) < 20:
        return None
    skip = int(round(equil_trials / TRIALS_PER_ROW))
    # a run appended to an earlier one restarts the CPU clock; fit the last run only
    reset = max((i for i in range(1, len(t)) if t[i] < t[i - 1]), default=0)
    if reset > skip:
        t, s = t[reset:], s[reset:]
        if len(t) < 20:
            return None
    xs, ys = [], []
    for i in range(skip, len(t)):
        if s[i] > 0 and t[i] > 0:
            xs.append(math.log(t[i]))
            ys.append(math.log(s[i]))
    if len(xs) < 10:
        return None
    return float(np.mean([y + 0.5 * x for x, y in zip(xs, ys)]))


def prod_cycles(prefix):
    """Number of production cycles: the last cycles_to_complete in the FEASST input."""
    for f in sorted(glob.glob(prefix + "*_fstin.txt")):
        vals = re.findall(r"cycles_to_complete=(\d+)", open(f, errors="replace").read())
        if vals:
            return int(vals[-1])
    return None


def num_accepted_at_production_end(fn, P):
    """Summed num_accepted at production row P of the last run in the log."""
    if not os.path.exists(fn):
        return None
    rows = list(csv.reader(open(fn)))
    if len(rows) < 2:
        return None
    idx = [i for i, h in enumerate(rows[0]) if h == "num_accepted"]
    if not idx:
        return None
    vals = []
    for r in rows[1:]:
        try:
            vals.append(sum(float(r[i]) for i in idx))
        except (ValueError, IndexError):
            vals.append(None)                 # repeated header of an appended run
    start = 0
    for k in range(1, len(vals)):
        if vals[k] is None:
            start = k + 1
        elif vals[k - 1] is not None and vals[k] < vals[k - 1]:
            start = k                         # counter reset: a new run began
    seg = [v for v in vals[start:] if v is not None]
    if not seg:
        return None
    return seg[P - 1] if (P and len(seg) >= P) else seg[-1]


def analyze(prefix, K=1):
    """R_acc, z12 (with SEMs) and product fraction for one run prefix with K sites."""
    P = prod_cycles(prefix)
    nacc = lambda fn: num_accepted_at_production_end(fn, P)
    ba = np.array([x for x in (nacc(f"{prefix}n0s{s}.csv") for s in range(0, 32, 2)) if x], float)
    ma = np.array([x for x in (nacc(f"{prefix}n0s{s}.csv") for s in range(1, 32, 2)) if x], float)
    bi = np.array([x for x in (intercept(prefix, s) for s in range(0, 32, 2)) if x is not None], float)
    mi = np.array([x for x in (intercept(prefix, s) for s in range(1, 32, 2)) if x is not None], float)
    n2 = [last_value(f"{prefix}n0s{s}_num2.csv", "average") for s in range(0, 32, 2)]
    n2 = np.array([x for x in n2 if x is not None], float)

    out = dict(R_acc=np.nan, R_err=np.nan, z12=np.nan, z_err=np.nan,
               prodfrac=np.nan, n_b=len(ba), n_m=len(ma))
    sem = lambda v: v.std(ddof=1) / math.sqrt(v.size) if v.size > 1 else 0.0
    if ba.size and ma.size:
        out["R_acc"] = ba.mean() / ma.mean()
        out["R_err"] = out["R_acc"] * math.hypot(sem(ba) / ba.mean(), sem(ma) / ma.mean())
    if bi.size and mi.size:
        out["z12"] = math.exp(2 * (mi.mean() - bi.mean()))
        out["z_err"] = out["z12"] * 2 * math.hypot(sem(bi), sem(mi))
    if n2.size:
        out["prodfrac"] = n2.mean() / K
    return out


SWEEPS = {
    "density": [(n, f"dens_flat_k1_n{n}_a1.05_chemi", 1) for n in (50, 150, 300, 450)],
    "system size": [(n, f"syssize_flat_k1_n{n}_a1.05_chemi", 1) for n in (50, 150, 300, 450, 600, 800)],
    "shell cutoff": [(a, f"avdist_flat_k1_n450_a{a}_chemi", 1) for a in ("1.1", "1.15", "1.2", "1.25", "1.3", "1.35")],
    "sites": [(k, f"multi_spread_k{k}_n450_a1.05_chemi", k) for k in (1, 2, 4, 6, 9)],
}

if __name__ == "__main__":
    os.chdir(sys.argv[1] if len(sys.argv) > 1 else ".")
    print(f"{'sweep':>12} {'x':>6} {'R_acc':>16} {'z12':>14}")
    for name, cfgs in SWEEPS.items():
        for x, prefix, K in cfgs:
            if glob.glob(prefix + "n0s0.csv"):
                r = analyze(prefix, K)
                print(f"{name:>12} {x!s:>6} {r['R_acc']:9.2f} ± {r['R_err']:5.2f} {r['z12']:7.1f} ± {r['z_err']:4.1f}")
