#!/usr/bin/env python3
"""HIGH-DIMENSIONAL relational queries: where the grid dies and the bundle does not.

The 1-D complexity verdict (cwf_fpe_complexity.py) was a tie: a histogram->FFT
computes the difference histogram as fast as the bundle readout.  But the FPE bundle
is the empirical characteristic function (ECF) of the stored multiset sampled at N
random frequencies, while the histogram is the ECF on a REGULAR GRID -- and a grid
over an m-dimensional value domain costs B^m bins.  Monte-Carlo frequency sampling
(random features) does not.  This script measures the split.

Task (same relational query in every dimension m): a PLANTED bundle stores k = 20
points of which 10 pairs share a displacement vector Delta; a NULL bundle stores 20
unpaired points at the same density.  Detection: is the difference-histogram value
at Delta larger for the planted set than for the null set?  (Well-posed at every m;
in m = 1 the accidental-pair background is dense, so both methods degrade there
alike.)  Methods: (a) the FPE bundle |b|^2 readout (N = 4000 complex components for
EVERY m), (b) SPARSE pairwise bin-coincidence from the raw points (the honest
small-k classical baseline: O(k^2 m) time, O(km) memory, any m), (c) a DENSE
B-bins-per-axis histogram via shifted dot products (exact, non-circular; cost and
memory ~ B^m).  The B^m wall applies to (c) only; at small k, (b) wins everything --
the m-separation over dense grids is real, but the bundle-vs-best-classical contest
at small k belongs to the sparse route (see the manuscript's rescoped claim).

Seeded, CPU.  Writes cwf_fpe_highdim_results.json.
"""
import json
import time

import numpy as np

N = 4000          # bundle dimension, fixed for all m
SIGMA = 2.0       # frequency bandwidth per axis (kernel HWHM ~ 0.59)
R = 10.0          # value range per axis
B = 48            # histogram bins per axis (width 0.21 << kernel HWHM)
K_PAIRS = 10      # planted pairs sharing the displacement Delta
TRIALS = 40
MEM_CAP = 2e9     # bytes; grids beyond this are reported as estimated-only


def hist_score(pts, delta, m):
    """Difference-histogram value at displacement delta from a B^m grid
    (shifted dot product; exact, non-circular)."""
    H = np.zeros((B,) * m)
    idx = np.clip((pts / R * B).astype(int), 0, B - 1)
    np.add.at(H, tuple(idx.T), 1.0)
    sh = np.clip((delta / R * B).round().astype(int), 0, B - 1)
    sl_a = tuple(slice(0, B - s) for s in sh)
    sl_b = tuple(slice(s, B) for s in sh)
    return float((H[sl_a] * H[sl_b]).sum())


def one_trial(m, rng):
    theta = rng.normal(0.0, SIGMA, (N, m))
    delta = rng.uniform(1.0, 3.0, m)                       # planted displacement
    base = rng.uniform(0.5, R - 3.5, (K_PAIRS, m))
    pts = np.vstack([base, base + delta])                  # planted: 2*K_PAIRS pts
    null = rng.uniform(0.5, R - 0.5, (2 * K_PAIRS, m))     # matched unpaired set

    # ---- (a) bundle readout ------------------------------------------------
    t0 = time.perf_counter()
    b = np.exp(1j * pts @ theta.T).sum(0)                  # bundle, O(kN)
    t_build = time.perf_counter() - t0
    b0 = np.exp(1j * null @ theta.T).sum(0)
    cosd = np.cos(theta @ delta)
    t0 = time.perf_counter()
    s_pl = float(cosd @ (np.abs(b) ** 2)) / N              # ECF autocorr at delta
    t_bundle = time.perf_counter() - t0
    s_nu = float(cosd @ (np.abs(b0) ** 2)) / N
    ok_b = bool(s_pl > s_nu)

    # ---- (b) SPARSE pairwise route (the honest small-k classical baseline):
    #      exact continuous pairwise comparison from the k raw points -- no
    #      binning at all.  O(k^2 m) time, O(k m) memory, any m. -------------
    def sparse_score(P):
        diff = P[:, None, :] - P[None, :, :]               # (k,k,m)
        return int((np.abs(diff - delta) < 0.3).all(-1).sum())
    t0 = time.perf_counter()
    s_sp = sparse_score(pts)
    t_sparse = time.perf_counter() - t0
    ok_s = bool(s_sp > sparse_score(null))
    sparse_mem = pts.nbytes

    # ---- (c) DENSE grid (histogram) route ----------------------------------
    mem = (B ** m) * 8.0                                   # float64 grid
    if mem > MEM_CAP:
        return dict(ok_b=ok_b, t_bundle=t_bundle, t_build=t_build,
                    ok_s=ok_s, t_sparse=t_sparse, sparse_mem=sparse_mem,
                    ok_g=None, t_grid=None, mem=mem)
    t0 = time.perf_counter()
    g_pl = hist_score(pts, delta, m)
    t_grid = time.perf_counter() - t0
    g_nu = hist_score(null, delta, m)
    ok_g = bool(g_pl > g_nu)
    return dict(ok_b=ok_b, t_bundle=t_bundle, t_build=t_build,
                ok_s=ok_s, t_sparse=t_sparse, sparse_mem=sparse_mem,
                ok_g=ok_g, t_grid=t_grid, mem=mem)


def main():
    out = {"params": dict(N=N, sigma=SIGMA, R=R, B=B, k=2 * K_PAIRS,
                          trials=TRIALS)}
    rows = []
    print(f"k={2*K_PAIRS} points, {K_PAIRS} pairs sharing Delta; N={N} for all m; "
          f"grid B={B} bins/axis")
    for m in [1, 2, 3, 4, 5, 6]:
        res = [one_trial(m, np.random.default_rng(100 * m + t))
               for t in range(TRIALS)]
        acc_b = float(np.mean([r["ok_b"] for r in res]))
        tb = float(np.median([r["t_bundle"] for r in res]))
        acc_s = float(np.mean([r["ok_s"] for r in res]))
        ts = float(np.median([r["t_sparse"] for r in res]))
        smem = res[0]["sparse_mem"]
        grid_ok = [r["ok_g"] for r in res if r["ok_g"] is not None]
        acc_g = float(np.mean(grid_ok)) if grid_ok else None
        tg = (float(np.median([r["t_grid"] for r in res if r["t_grid"]]))
              if grid_ok else None)
        mem = res[0]["mem"]
        rows.append(dict(m=m, acc_bundle=acc_b, t_bundle_ms=tb * 1e3,
                         acc_sparse=acc_s, t_sparse_ms=ts * 1e3,
                         sparse_mem_bytes=smem,
                         acc_grid=acc_g, t_grid_ms=(tg * 1e3 if tg else None),
                         grid_mem_bytes=mem, grid_feasible=bool(grid_ok)))
        gm = f"{mem/1e6:.1f} MB" if mem < 1e9 else f"{mem/1e9:.1f} GB"
        gs = (f"dense grid acc={acc_g:.2f} t={tg*1e3:.2f} ms" if grid_ok
              else "dense grid > mem cap")
        print(f"  m={m}: bundle acc={acc_b:.2f} t={tb*1e3:.2f} ms  |  "
              f"sparse acc={acc_s:.2f} t={ts*1e3:.3f} ms mem={smem} B  |  "
              f"{gs}  mem={gm}")
    out["rows"] = rows
    with open("cwf_fpe_highdim_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote cwf_fpe_highdim_results.json")


if __name__ == "__main__":
    main()
