#!/usr/bin/env python3
"""Fig. 8 data: distances from the acid-site oxygen in MFI (50w, mu=-550 kJ/mol),
from the 32 trajectories acid_zeon0s*_eq.xyz (mobile group) in the folder given.
Site types: 0 = O of framework-OH, 2 = water O, 5 = O- (deprotonated site), 7 = H3O+ O.
Writes mfi_distances.npz with, per frame (minimum-image distances):
  r_h3o  : reacted frames,   r(O-, O_H3O+)             -> Fig. 8 (red)
  r_pnear: reacted frames,   r(O-, nearest water O)    -> Fig. 8 (blue)
  r_pall : reacted frames,   r(O-, every water O)
  r_wnear: unreacted frames, r(O_OH, nearest water O)
  r_wall : unreacted frames, r(O_OH, every water O)
Usage: python mfi_distances.py MFI_RUNS/50w/mu_550"""
import sys, glob
import numpy as np

RUN = sys.argv[1]
r_h3o, r_wnear, r_wall, r_pnear, r_pall = [], [], [], [], []
nframe = nreact = 0
for path in sorted(glob.glob(f"{RUN}/acid_zeon0s*_eq.xyz")):
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            n = int(line)
            box = np.array(f.readline().split()[1:4], float)
            st = np.empty(n, int); co = np.empty((n, 3))
            for i in range(n):
                p = f.readline().split()
                st[i] = int(p[0]); co[i] = p[1:4]
            nframe += 1
            def d(a, b):
                v = co[b] - co[a]
                v -= box * np.round(v / box)
                return np.linalg.norm(v, axis=-1)
            o5, o7, o0 = np.flatnonzero(st == 5), np.flatnonzero(st == 7), np.flatnonzero(st == 0)
            if o5.size and o7.size:
                r_h3o.append(d(o5[0], o7[0])); nreact += 1
                rw = d(o5[0], np.flatnonzero(st == 2))
                r_pnear.append(rw.min()); r_pall.extend(rw)
            elif o0.size:
                rw = d(o0[0], np.flatnonzero(st == 2))
                r_wnear.append(rw.min()); r_wall.extend(rw)
np.savez("mfi_distances.npz", r_h3o=np.array(r_h3o), r_wnear=np.array(r_wnear),
         r_wall=np.array(r_wall), r_pnear=np.array(r_pnear), r_pall=np.array(r_pall), box=box)
print(f"frames {nframe}, reacted {nreact}, unreacted {len(r_wnear)}, box {box}")
for k, v in [("H3O+ (reacted)", r_h3o), ("nearest water (unreacted)", r_wnear), ("all water (unreacted)", r_wall),
             ("nearest water to O- (reacted)", r_pnear), ("all water to O- (reacted)", r_pall)]:
    v = np.array(v)
    if v.size == 0:
        print(f"{k:28s} n=      0  (no frames in this state)")
        continue
    print(f"{k:28s} n={v.size:7d} median {np.median(v):.2f}  frac 2.4-3.7: {((v>=2.4)&(v<=3.7)).mean():.3f}"
          f"  frac > Lmin/2 ({box.min()/2:.2f}): {(v>box.min()/2).mean():.3f}  max {v.max():.2f}")
