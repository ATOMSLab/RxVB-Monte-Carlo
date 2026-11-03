"""Reaction-volume occupancy in the slit pore (Table III, slit rows).

Counts B particles (type 1) within 1.00-1.05 sigma of the site A (type 0) in the
reactant-state frames of each replica's trajectory, and reports the mean over
replicas of <N | N >= 1>, with its SEM and the fraction of occupied frames.

Usage: python slit_occupancy.py RUN_DIR .xyz 10   (production trajectories, every 10th frame)"""
import sys, glob, os, numpy as np
D, SUF, STRIDE = sys.argv[1], sys.argv[2], int(sys.argv[3])
def rep(fn):
    L = open(fn).read().split('\n')
    n = int(L[0]); box = np.array([float(v) for v in L[1].split()[1:4]])
    nf = len(L) // (n + 2); cnt = []
    for k in range(0, nf, STRIDE):
        blk = L[k*(n+2)+2: k*(n+2)+2+n]
        if len(blk) < n or not blk[-1].strip(): continue
        a = np.array([l.split() for l in blk], float)
        t, x = a[:, 0].astype(int), a[:, 1:4]
        A = x[t == 0]
        if len(A) == 0: continue                       # product state
        d = x[t == 1] - A[0]; d -= box * np.round(d / box)
        r = np.sqrt((d*d).sum(1)); cnt.append(int(((r >= 1.0) & (r <= 1.05)).sum()))
    c = np.array(cnt, float); occ = c[c >= 1]
    return occ.mean() if len(occ) else np.nan, (c >= 1).mean(), len(c)
res = {}
for f in sorted(glob.glob(os.path.join(D, 'chemn0s*' + SUF))):
    s = int(os.path.basename(f).split('n0s')[1].split('.')[0].split('_')[0])
    if SUF == '.xyz' and f.endswith('_eq.xyz'): continue
    res[s] = rep(f)
def summ(keys):
    v = np.array([res[k][0] for k in keys if np.isfinite(res[k][0])])
    return v.mean(), v.std(ddof=1)/np.sqrt(len(v)), len(v), int(sum(res[k][2] for k in keys))
allk = sorted(res); ev = [k for k in allk if k % 2 == 0]; od = [k for k in allk if k % 2]
for lab, ks in (('all', allk), ('even', ev), ('odd', od)):
    m, e, n, nfr = summ(ks); print(f"{D} {SUF} {lab:4s}: <n|n>=1> = {m:.4f} +- {e:.4f}  (reps {n}, reactant frames {nfr})  P_occ={np.mean([res[k][1] for k in ks]):.3f}")
