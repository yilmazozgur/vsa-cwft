#!/usr/bin/env python3
"""RELATIONAL READOUT BEYOND ITEM-RECOVERY CAPACITY (future-work route iii, tested).

Can the difference histogram of a bundle be read at loads where the items
themselves can no longer be recovered?  Design mirrors the holding-capacity
experiment (cwf_fpe_uncertainty.py part3): L = 4N well-separated grid cells
(sigma=4, spacing=3, K(spacing)~0), k of them stored, item readout by thresholded
similarity (sim > 0.5).  The planted relation: the k stored cells form k/2 pairs
separated by DLAG = 3 cells (Delta = 9.0).  Three curves vs load k:

  1. item recall    -- fraction of stored cells with sim > 0.5 (plus the FP rate
                       among the L-k non-stored cells);
  2. decode-route   -- threshold-decode the cell list (with its false positives),
                       then difference it: does lag DLAG win among lag multiples?
  3. direct spectral -- the |b|^2 autocorrelation readout (no decoding): does the
                       score at DLAG win among lag multiples 1..8 (excluding 0)?

Item capacity is ~0.07N; the question is whether curve 3 outlives curve 1.
Seeded, CPU.  Writes cwf_fpe_beyond_capacity_results.json.
"""
import json

import numpy as np

N = 1000
SIGMA = 4.0
SPACING = 3.0
DLAG = 3                    # planted pair displacement, in cells (Delta = 9.0)
LAG_CANDS = np.arange(1, 9)  # candidate lags (cells) for detection
TRIALS = 20
KS = [20, 40, 70, 140, 300, 600, 1000, 1600, 2400]


def one_trial(k, rng):
    th = SIGMA * rng.standard_normal(N)
    L = 4 * N
    xs = np.arange(L) * SPACING
    n_pair = k // 2
    base = rng.choice(L - DLAG, size=n_pair, replace=False)
    cells = np.unique(np.concatenate([base, base + DLAG]))
    Phi = np.exp(1j * np.outer(xs[cells], th))
    b = Phi.sum(0)

    # 1) item recall / false positives at the sim > 0.5 threshold
    PhiL = np.exp(1j * np.outer(xs, th))
    sim = np.real(PhiL @ b.conj()) / N
    pred = sim > 0.5
    true = np.zeros(L, bool)
    true[cells] = True
    recall = float((pred & true).sum() / len(cells))
    fp = float((pred & ~true).sum() / max(1, (~true).sum()))

    # 2) decode-route: difference histogram of the decoded (thresholded) cell list
    dec_cells = np.where(pred)[0]
    ok_decode = False
    if len(dec_cells) >= 2:
        diffs = np.abs(dec_cells[None, :] - dec_cells[:, None])
        counts = [(diffs == lag).sum() for lag in LAG_CANDS]
        ok_decode = bool(LAG_CANDS[int(np.argmax(counts))] == DLAG)

    # 3) direct spectral readout at the candidate lags (no decoding)
    score = (np.cos(np.outer(LAG_CANDS * SPACING, th)) @ (np.abs(b) ** 2)) / N
    ok_direct = bool(LAG_CANDS[int(np.argmax(score))] == DLAG)

    return recall, fp, ok_decode, ok_direct


def main():
    out = {"params": dict(N=N, sigma=SIGMA, spacing=SPACING, dlag=DLAG,
                          L=4 * N, trials=TRIALS, capacity_007N=0.07 * N)}
    rows = []
    print(f"N={N} (item capacity ~ {0.07*N:.0f}), k/2 pairs at lag {DLAG} cells")
    for k in KS:
        res = [one_trial(k, np.random.default_rng(1000 + 31 * k + t))
               for t in range(TRIALS)]
        recall = float(np.mean([r[0] for r in res]))
        fp = float(np.mean([r[1] for r in res]))
        dec = float(np.mean([r[2] for r in res]))
        direct = float(np.mean([r[3] for r in res]))
        rows.append(dict(k=k, recall=recall, fp=fp, decode=dec, direct=direct))
        print(f"  k={k:>4}: recall={recall:.2f} (fp={fp:.3f})  "
              f"decode-route={dec:.2f}  direct spectral={direct:.2f}")
    out["rows"] = rows
    with open("cwf_fpe_beyond_capacity_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote cwf_fpe_beyond_capacity_results.json")


if __name__ == "__main__":
    main()
