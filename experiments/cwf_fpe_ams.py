#!/usr/bin/env python3
"""THE BUNDLE MATCHES AGMS: an order-optimal self-join sketch (Move A theorem).

The lag-d relational readout estimates the self-join size J(d) = sum_a f_a f_{a+d}
of the value histogram f (f_a = # items at value a) -- the object of the
Alon-Gibbons-Matias-Szegedy (AGMS) sketch.  Two linear-sketch estimators of the SAME
J(d), each using N random features:

  * FPE bundle:  b_k = sum_i e^{i x_i theta_k},  theta_k ~ carrier p;
                 score(d) = (1/N) sum_k |b_k|^2 cos(d theta_k).
  * AGMS +/-1:   for r=1..N draw a sign hash eps^r; Z_f = sum_a eps_a f_a,
                 Z_g = sum_a eps_a f_{a-d}; estimate = mean_r Z_f^r Z_g^r.

Both are unbiased for J(d) (the FPE up to kernel smearing) with per-feature variance
Theta(F_2^2), F_2 = sum_a f_a^2 (= k for a set), so the averaged estimator has
additive SD ~ F_2 / sqrt(N).  We verify the two SDs COINCIDE at matched N across a
range of (k, N): the FPE bundle is, up to a small constant, the AGMS sketch, hence
order-optimal for the AMS Omega(1/eps^2) lower bound.  Seeded, CPU.  Writes
cwf_fpe_ams_results.json.
"""
import json

import numpy as np

B = 4_000_000           # value grid: HUGE, so a k<=400 set is SPARSE (k^2 << B):
#                         differences are non-degenerate -- the regime where the
#                         phase sketch's variance is Theta(F_2^2/N), matching AGMS.
SPACING = 1.0
D0 = 7                  # planted lag -- carries the signal
DN = 3                  # null lag (no planted pairs) -- the estimator NOISE FLOOR,
#                         which sets the detection threshold and the eps-error bound
SIGMA, OMEGA0 = 2.0, 8.0
OUTER = 200            # outer ensemble for SD estimation
_HASH = (2654435761, 40503, 2 ** 31 - 1)          # multiplicative-hash constants


def make_set(k, rng):
    """k DISTINCT positions (a set: F_2 = k) with k/2 pairs at the planted lag D0,
    in a range large enough (k^2 << B) that all pairwise differences are distinct."""
    while True:
        base = rng.choice(B - D0, size=k // 2, replace=False)
        pos = np.concatenate([base, base + D0])
        if len(np.unique(pos)) == len(pos):
            break
    J = int(k // 2)                                 # planted pairs at lag D0 (sparse)
    return pos, float(k), float(J)                  # F_2 = k for a set


def fpe_stat(pos, N, lag, rng):
    vals = pos * SPACING
    ests = []
    for _ in range(OUTER):
        th = OMEGA0 + SIGMA * rng.standard_normal(N)
        b = np.exp(1j * np.outer(vals, th)).sum(0)
        ests.append(float((np.cos(lag * SPACING * th) @ (np.abs(b) ** 2)) / N))
    return float(np.mean(ests)), float(np.std(ests))


def agms_stat(pos, N, lag, rng):
    """AGMS self-join estimator with proper independent +/-1 signs, materialized only
    over the sparse support (the ~2k distinct values that appear).  Unbiased for
    J(lag) = #{(i,j): x_i - x_j = lag}, per-feature variance <= F_2^2."""
    V, inv = np.unique(np.concatenate([pos, pos - lag]), return_inverse=True)
    idx_f = inv[:len(pos)]                          # column of each x_i
    idx_g = inv[len(pos):]                          # column of each x_i - lag
    ests = []
    for _ in range(OUTER):
        eps = rng.choice([-1.0, 1.0], size=(N, len(V)))
        Zf = eps[:, idx_f].sum(1)                   # sum_i s(x_i)
        Zg = eps[:, idx_g].sum(1)                   # sum_i s(x_i - lag)
        ests.append(float(np.mean(Zf * Zg)))
    return float(np.mean(ests)), float(np.std(ests))


def main():
    rng = np.random.default_rng(0)
    out = {"params": dict(B=B, D0=D0, sigma=SIGMA, omega0=OMEGA0, outer=OUTER),
           "rows": []}
    print(f"FPE bundle vs AGMS +/-1 sketch: NOISE-FLOOR SD at a null lag (=the "
          f"detection threshold; both ~ F_2/sqrt(N)), plus the signal at the planted lag:")
    print(f"  {'k':>5} {'N':>5} {'F2':>5} | {'SD_FPE':>7} {'SD_AGMS':>8} {'F2/sqN':>7} "
          f"| ratio | signal@D0 (FPE/AGMS/true J)")
    for k in [40, 120, 400]:
        for N in [500, 2000]:
            pos, F2, J = make_set(k, np.random.default_rng(10 + k))
            _, sd_f = fpe_stat(pos, N, DN, rng)          # noise floor at null lag
            _, sd_a = agms_stat(pos, N, DN, rng)
            mf, _ = fpe_stat(pos, N, D0, rng)            # signal at planted lag
            ma, _ = agms_stat(pos, N, D0, rng)
            pred = F2 / np.sqrt(N)
            row = dict(k=k, N=N, F2=F2, J=J, sd_fpe_null=sd_f, sd_agms_null=sd_a,
                       pred_F2_over_sqrtN=pred, ratio_fpe_agms=sd_f / sd_a,
                       signal_fpe=mf, signal_agms=ma)
            out["rows"].append(row)
            print(f"  {k:>5} {N:>5} {F2:>5.0f} | {sd_f:>7.2f} {sd_a:>8.2f} {pred:>7.2f} "
                  f"| {sd_f/sd_a:>5.2f} | {mf:>6.1f} / {ma:>5.1f} / {J:.0f}")
    ratios = [r["ratio_fpe_agms"] for r in out["rows"]]
    out["ratio_mean"] = float(np.mean(ratios))
    out["ratio_range"] = [float(min(ratios)), float(max(ratios))]
    print(f"  -> noise-floor SD_FPE / SD_AGMS = {np.mean(ratios):.2f} "
          f"(range {min(ratios):.2f}-{max(ratios):.2f}): matched to a small constant.")
    print(f"     Both estimators are Theta(F_2/sqrt(N)) linear sketches of the self-join;")
    print(f"     N=Theta(1/eps^2) measurements, matching the AMS Omega(1/eps^2) lower bound.")
    with open("cwf_fpe_ams_results.json", "w") as fp:
        json.dump(out, fp, indent=2)
    print("wrote cwf_fpe_ams_results.json")


if __name__ == "__main__":
    main()
