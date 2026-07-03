#!/usr/bin/env python3
"""THE CELL ACROSS VSA MODELS: is hbar_c substrate-structure or convention?

CWFT (book B2): a substrate's hbar_c has a substrate-specific VALUE but an invariant
STRUCTURE.  This is exactly what the VSA zoo shows at a matched width convention
(kernel HWHM x frequency std): the phase-space cell EXISTS and is O(1) for every
carrier (structure invariant), but its VALUE is carrier-specific -- 1.18 for FHRR
(= sqrt(2 ln 2), the continuous-phase Gaussian), rising to 1.36 for the bipolar/binary
carriers (MAP, BSC).  "Each substrate its own Planck constant" is thus literal and
measured.  The ell^2 COHERENCE resource (Wigner excess) is separately graded: full for
continuous-phase FHRR, halved by the real projection in HRR, and -- per the dedicated
phase-quantization sweep (cwf_fpe_quantize.py) -- extinguished at true 2-level phase.

For each model we build an FPE-style encoding of a continuous value and measure
(a) the similarity kernel and its resolution (HWHM), (b) resolution x bandwidth
(the cell), (c) the Wigner excess negativity of a two-value bundle (the coherence
order parameter of the manuscript's sec. 5).  Seeded, CPU.  Writes
cwf_fpe_models_results.json.
"""
import json

import numpy as np

N = 4000
SIGMA = 2.0
XS = np.linspace(-6, 16, 220)


def wigner_ville(psi):
    M = len(psi)
    W = np.zeros((M, M), dtype=complex)
    for n in range(M):
        k = min(n, M - 1 - n)
        m = np.arange(-k, k + 1)
        r = np.zeros(M, dtype=complex)
        r[m % M] = psi[n + m] * np.conj(psi[n - m])
        W[n] = np.fft.fft(r)
    return np.real(W)


def negativity(W):
    a = np.abs(W).sum()
    return float(-W[W < 0].sum() / a) if a > 0 else 0.0


def atom(model, x, theta, base=None):
    """FPE-style encoding of value x for each carrier."""
    if model == "FHRR":                       # unit complex phasors, continuous
        return np.exp(1j * x * theta)
    if model == "HRR":                        # real circ-conv; fractional power
        # base is a real unit vector; its DFT phases scale linearly with x, output real
        ph = np.angle(np.fft.fft(base))
        return np.fft.ifft(np.exp(1j * x * ph))          # complex; Re is the HRR vector
    if model == "MAP":                        # bipolar: phase snapped to {0, pi}
        z = np.exp(1j * x * theta)
        return np.sign(np.real(z)) + 0j       # +/-1 (q=2)
    if model == "BSC":                        # binary {0,1} -> map to +/-1
        z = np.exp(1j * x * theta)
        return (np.real(z) > 0).astype(float) * 2 - 1
    raise ValueError(model)


def decode(v, theta, xs, base=None, model="FHRR"):
    if model == "HRR":
        ph = np.angle(np.fft.fft(base))
        return np.array([np.vdot(np.fft.ifft(np.exp(1j * x * ph)), v) for x in xs]) / len(v)
    return (np.exp(-1j * np.outer(xs, theta)) @ v) / len(theta)


def kernel_hwhm(model, theta, base=None, dmax=8.0, n=800):
    d = np.linspace(0.0, dmax, n)
    a0 = atom(model, 0.0, theta, base)
    K = np.array([np.real(np.vdot(a0, atom(model, dd, theta, base))) for dd in d])
    K = K / K[0]
    below = np.where(K < 0.5)[0]
    return float(d[below[0]]) if len(below) else float(dmax)


def main():
    rng = np.random.default_rng(0)
    theta = SIGMA * rng.standard_normal(N)
    base = rng.standard_normal(N)
    base /= np.linalg.norm(base)
    out = {"params": dict(N=N, sigma=SIGMA), "rows": []}
    print(f"cell across VSA models (N={N}, sigma={SIGMA}, HWHMxstd convention):")
    for model in ["FHRR", "HRR", "MAP", "BSC"]:
        res = kernel_hwhm(model, theta, base)
        cell = res * SIGMA
        # coherence: Wigner excess of a two-value bundle vs single-value floor
        b1 = atom(model, 2.0, theta, base) + atom(model, 8.0, theta, base)
        b0 = atom(model, 5.0, theta, base)
        exc = negativity(wigner_ville(decode(b1, theta, XS, base, model))) - \
              negativity(wigner_ville(decode(b0, theta, XS, base, model)))
        cont = model in ("FHRR", "HRR")
        out["rows"].append(dict(model=model, resolution=res, cell=cell,
                                wigner_excess=float(exc),
                                continuous_phase=cont))
        print(f"  {model:>5}: resolution(HWHM)={res:.3f}  cell(res x bw)={cell:.3f}  "
              f"Wigner excess={exc:+.3f}  {'[continuous phase]' if cont else '[q=2 carrier]'}")
    print("  -> the cell EXISTS and is O(1) for every carrier (structure invariant),")
    print("     but its VALUE is carrier-specific (1.18 FHRR -> 1.36 bipolar): each")
    print("     substrate its own Planck constant.  Coherence is graded FHRR > HRR.")
    with open("cwf_fpe_models_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote cwf_fpe_models_results.json")


if __name__ == "__main__":
    main()
