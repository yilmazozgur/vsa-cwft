#!/usr/bin/env python3
"""A DOWNSTREAM TASK for squeezed FPE: value decoding under channel noise.

Encode a value x with flat vs squeezed (foveated) FPE at matched dimension and
bandwidth, corrupt the hypervector with complex Gaussian channel noise, decode by
matched-filter argmax over a fine grid, and report RMSE(x_hat) separately for
values inside the fovea (|x-5| < 1) and in the periphery (x in [1,3] u [7,9]).

Prediction: at matched budget, squeezing buys fovea precision at the price of
peripheral precision -- a measured, usable trade, not just a resolution profile.
200 trials/cell, 3 codebook seeds.  Writes cwf_fpe_fovea_task_results.json.
"""
import json

import numpy as np

N = 4000
SIGMA = 2.0
A, B, XF = 0.0, 10.0, 5.0
NOISE = 1.0                 # channel noise std per complex component
TRIALS = 200
SEEDS = 3
GRID = np.linspace(A, B, 2001)


def warp():
    xg = np.linspace(A, B, 4000)
    dens = 0.2 + np.exp(-0.5 * ((xg - XF) / 1.0) ** 2)
    cdf = np.cumsum(dens)
    cdf -= cdf[0]
    cdf /= cdf[-1]
    return xg, A + (B - A) * cdf


def run(seed):
    rng = np.random.default_rng(seed)
    th = SIGMA * rng.standard_normal(N)
    xg, wg = warp()
    wgrid = np.interp(GRID, xg, wg)
    enc = {"flat": lambda x: np.exp(1j * x * th),
           "squeezed": lambda x: np.exp(1j * np.interp(x, xg, wg) * th)}
    dec_book = {"flat": np.exp(-1j * np.outer(GRID, th)),
                "squeezed": np.exp(-1j * np.outer(wgrid, th))}
    out = {}
    for region, lo, hi in [("fovea", XF - 1, XF + 1), ("periphery", 1.0, 3.0)]:
        xs_true = rng.uniform(lo, hi, TRIALS)
        for kind in ("flat", "squeezed"):
            err = []
            for x in xs_true:
                v = enc[kind](x) + NOISE * (rng.standard_normal(N)
                                            + 1j * rng.standard_normal(N)) / np.sqrt(2)
                sim = np.real(dec_book[kind] @ v)
                err.append(GRID[int(np.argmax(sim))] - x)
            out[(region, kind)] = float(np.sqrt(np.mean(np.square(err))))
    return out


def main():
    res = [run(s) for s in range(SEEDS)]
    rows = []
    print(f"decoding RMSE under channel noise (std {NOISE}/component), "
          f"{TRIALS} trials x {SEEDS} seeds:")
    for region in ("fovea", "periphery"):
        for kind in ("flat", "squeezed"):
            vals = [r[(region, kind)] for r in res]
            rows.append(dict(region=region, encoding=kind,
                             rmse_mean=float(np.mean(vals)),
                             rmse_std=float(np.std(vals))))
            print(f"  {region:>9} / {kind:>8}: RMSE = "
                  f"{np.mean(vals):.4f} +/- {np.std(vals):.4f}")
    out = dict(params=dict(N=N, sigma=SIGMA, fovea=XF, noise=NOISE,
                           trials=TRIALS, seeds=SEEDS), rows=rows)
    with open("cwf_fpe_fovea_task_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote cwf_fpe_fovea_task_results.json")


if __name__ == "__main__":
    main()
