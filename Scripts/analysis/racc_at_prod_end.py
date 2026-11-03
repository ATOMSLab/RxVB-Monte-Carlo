#!/usr/bin/env python3
"""R_acc for every slit-pore run prefix in a folder, with N_acc at the end of production.

Each input ends with WriteFileAndCheck and Run until_file_exists, so a replica
that finishes early keeps running, and accepting reactions, until all replicas
are done. Reading the last log row therefore overcounts the unbiased replicas,
which finish first. This script reads the summed num_accepted at production row
P (the last cycles_to_complete in the *_fstin.txt, one log row per cycle) and
reports R_acc from both the last row (racc_old) and row P (racc_new).
Even sims are RxVB, odd sims unbiased.

Usage: python racc_at_prod_end.py [FOLDER]   (writes racc_prodend.csv)
"""
import csv, glob, math, os, re, statistics as st, sys

D = sys.argv[1] if len(sys.argv) > 1 else '.'


def prod_cycles(prefix):
    for f in sorted(glob.glob(os.path.join(D, prefix + '*_fstin.txt'))):
        vals = re.findall(r'cycles_to_complete=(\d+)', open(f, errors='replace').read())
        if vals:
            return int(vals[-1])            # the last Metropolis block is production
    return None


def last_run_counts(fn, P):
    """(count at last row, count at production row P, rows in the last run) for the summed num_accepted."""
    if not os.path.exists(fn):
        return None                         # fewer than 32 replicas in this folder
    rows = list(csv.reader(open(fn, errors='replace')))
    if len(rows) < 2:
        return None
    idx = [i for i, h in enumerate(rows[0]) if h == 'num_accepted']
    if not idx:
        return None
    vals = []
    for r in rows[1:]:
        try:
            vals.append(sum(float(r[i]) for i in idx))
        except (ValueError, IndexError):
            vals.append(None)               # repeated header of an appended run
    start = 0
    for k in range(1, len(vals)):
        if vals[k] is None:
            start = k + 1
        elif vals[k - 1] is not None and vals[k] < vals[k - 1]:
            start = k                       # counter reset: a new run began
    seg = [v for v in vals[start:] if v is not None]
    if not seg:
        return None
    at_end = seg[P - 1] if (P and len(seg) >= P) else seg[-1]
    return seg[-1], at_end, len(seg)


def ratio(b, m):
    mb, mm = st.mean(b), st.mean(m)
    sb = st.stdev(b) / math.sqrt(len(b)) if len(b) > 1 else 0.0
    sm = st.stdev(m) / math.sqrt(len(m)) if len(m) > 1 else 0.0
    r = mb / mm
    return r, r * math.hypot(sb / mb, sm / mm)


def main():
    prefixes = sorted({os.path.basename(f).split('n0s0.csv')[0] for f in glob.glob(os.path.join(D, '*_chemin0s0.csv'))})
    out = []
    print(f'{"prefix":42s} {"P":>6s} {"rows RxVB":>9s} {"rows unb":>9s} {"R_acc old (last row)":>22s} {"R_acc new (end of prod)":>24s} {"new/old":>8s}')
    for p in prefixes:
        P = prod_cycles(p)
        res = [last_run_counts(os.path.join(D, f'{p}n0s{s}.csv'), P) for s in range(32)]
        b = [r for r in res[0::2] if r]; m = [r for r in res[1::2] if r]
        if len(b) < 2 or len(m) < 2:
            print(f'{p:42s}  (incomplete: {len(b)} RxVB, {len(m)} unbiased logs)'); continue
        r_old, e_old = ratio([x[0] for x in b], [x[0] for x in m])
        r_new, e_new = ratio([x[1] for x in b], [x[1] for x in m])
        rb, rm = st.mean(x[2] for x in b), st.mean(x[2] for x in m)
        short = sum(1 for x in b + m if P and x[2] < P)
        print(f'{p:42s} {P if P else -1:6d} {rb:9.0f} {rm:9.0f} {r_old:14.2f} +- {e_old:5.2f} {r_new:16.2f} +- {e_new:5.2f} {r_new / r_old:8.3f}'
              + (f'   WARNING: {short} logs shorter than P' if short else ''))
        out.append((p, P, rb, rm, r_old, e_old, r_new, e_new))
    with open('racc_prodend.csv', 'w') as fh:
        fh.write('prefix,production_cycles,mean_rows_rxvb,mean_rows_unbiased,racc_old,racc_old_sem,racc_new,racc_new_sem\n')
        for r in out:
            fh.write(','.join(str(x) if isinstance(x, str) else f'{x:.6g}' for x in r) + '\n')
    print('wrote racc_prodend.csv')


if __name__ == '__main__':
    main()
