"""
cwf_fpe_wigner.py -- Wigner negativity as a non-classicality witness for VSA
bundles.  (VSA_CWFT_NOTE.md sec.5c, the strongest pay candidate of the FPE
phase-space program.)

A VSA vector v decodes to an amplitude profile over value-space,
    psi_v(x) = (1/N) sum_k v_k exp(-i x theta_k),
the "wavefunction in the x-basis" (theta_k = positive random frequencies -> psi is
a complex analytic signal).  A single FPE value phi(x0) -> psi = a bump at x0; a
BUNDLE phi(x1)+phi(x2) -> two bumps -- a coherent superposition (a computational
"cat state").  The Wigner-Ville distribution W(x,p) of psi is real but goes
NEGATIVE for cat states (interference fringes between the bumps).

The claim under test: Wigner negativity is a non-classicality WITNESS for VSA --
  * a single value is classical (W >= 0, negativity ~ 0);
  * a coherent bundle of separated values has W < 0 (genuine superposition);
  * a classical MIXTURE of the same values (phase-randomized ensemble average)
    has W >= 0 (negativity ~ 0) -- so the negativity distinguishes coherent
    (l2, amplitude) superposition from a classical (l1, probability) belief state.
  * negativity ONSETS at the resolution scale (fringes need separated bumps).
This is the measurable order parameter that the ground (l1) reading lacks --
exactly why Furlong-Eliasmith (2022) found negative fractional-binding quasi-
probability and DECLINED the Born rule; the negativity is why l1 is incomplete.

HONEST CAVEAT: that ANY coherent superposition of separated bumps has negative
Wigner is a math fact; the content is (a) VSA bundling realizes such superpositions
(so VSA states carry a genuine non-classicality witness), (b) it is a clean order
parameter (vs separation, vs decoherence), (c) decoherence/mixture kills it -- the
l2-vs-l1 split, measurable.  CPU, numpy, seeded.  Writes cwf_fpe_wigner_results.json.
"""

import json
import numpy as np

RNG = np.random.default_rng(0)


def make_freqs(N, omega0=4.0, sigma=1.0):
    """Positive random frequencies (analytic signal): theta ~ omega0 + sigma*Normal,
    kept > 0.  omega0 = carrier, sigma = bandwidth (sets resolution ~ 1/sigma)."""
    th = omega0 + sigma * RNG.standard_normal(N)
    return th[th > 0]


def fpe(x, theta):
    return np.exp(1j * x * theta)                       # FPE encoding of scalar x: (N,)


def profile(v, theta, xs):
    """Decoded wavefunction psi_v(x) over the value grid xs."""
    return (np.exp(-1j * np.outer(xs, theta)) @ v) / len(theta)


def wigner_ville(psi):
    """Discrete Wigner-Ville distribution W(x,p) of complex psi (real-valued)."""
    M = len(psi)
    W = np.zeros((M, M), dtype=complex)
    for n in range(M):
        kmax = min(n, M - 1 - n)
        r = np.zeros(M, dtype=complex)
        m = np.arange(-kmax, kmax + 1)
        r[m % M] = psi[n + m] * np.conj(psi[n - m])
        W[n] = np.fft.fft(r)
    return np.real(W)                                    # (x, p)


def negativity(W):
    """Fraction of total Wigner mass that is negative, in [0, 0.5]."""
    a = np.abs(W).sum()
    return float(-W[W < 0].sum() / a) if a > 0 else 0.0


def bundle_profile(centers, theta, xs, phases=None):
    """psi for a bundle sum_i e^{i phase_i} phi(x_i)."""
    if phases is None:
        phases = np.zeros(len(centers))
    v = sum(np.exp(1j * ph) * fpe(c, theta) for c, ph in zip(centers, phases))
    return profile(v, theta, xs)


if __name__ == "__main__":
    print("=" * 74)
    print("WIGNER NEGATIVITY as a non-classicality witness for VSA bundles")
    print("=" * 74)
    N = 3000
    theta = make_freqs(N)
    xs = np.linspace(-6, 16, 220)
    dx = xs[1] - xs[0]
    print(f"  N={len(theta)} positive freqs (carrier 4, bw 1); x-grid {len(xs)} pts, "
          f"dx={dx:.3f}; resolution ~ 1/bw = 1.0")

    # ---- baseline: single value = the CLASSICAL floor (WVD finite-N/sidelobe
    #      artifact, NOT genuine negativity).  Average over freq draws. ----
    def avg_neg(make_psi, ntr=6):
        vals = []
        for _ in range(ntr):
            th = make_freqs(N)
            vals.append(negativity(wigner_ville(make_psi(th))))
        return float(np.mean(vals)), float(np.std(vals))

    out = {}
    base, base_sd = avg_neg(lambda th: profile(fpe(5.0, th), th, xs))
    two, _ = avg_neg(lambda th: bundle_profile([2.0, 8.0], th, xs))
    print(f"\n[baseline] single value (classical floor) = {base:.4f} +/- {base_sd:.4f}")
    print(f"           two separated values            = {two:.4f}  "
          f"(genuine excess = {two - base:+.4f})")
    print(f"  -> the ~{base:.3f} floor is a WVD artifact; GENUINE negativity = excess above it.")
    out["baseline"] = {"single": base, "two": two, "excess": two - base}

    # ---- Part 1: negativity vs separation (onset at the resolution scale) ----
    print(f"\n[1] negativity vs separation Delta (resolution ~ 1.0); excess above floor:")
    P1 = []
    for D in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]:
        neg, _ = avg_neg(lambda th: bundle_profile([5 - D / 2, 5 + D / 2], th, xs))
        P1.append(dict(sep=D, neg=neg, excess=neg - base))
        print(f"    Delta={D:>4}: negativity={neg:.4f}  excess={neg - base:+.4f}")
    out["part1_separation"] = P1

    # ---- Part 2: coherent vs decohered mixture (the witness) ----
    print(f"\n[2] coherent superposition vs decohered MIXTURE (phase-randomized):")
    centers = [2.0, 8.0]
    neg_coh = negativity(wigner_ville(bundle_profile(centers, theta, xs)))
    Wmix = np.zeros((len(xs), len(xs)))
    NT = 300
    for _ in range(NT):
        ph = RNG.uniform(0, 2 * np.pi, len(centers))
        Wmix += wigner_ville(bundle_profile(centers, theta, xs, phases=ph))
    Wmix /= NT
    neg_mix = negativity(Wmix)
    # the witness: how much of the GENUINE (excess-above-floor) negativity does
    # decoherence remove?
    removed = (neg_coh - neg_mix) / (neg_coh - base + 1e-12)
    print(f"    coherent bundle:          negativity = {neg_coh:.4f}  (excess {neg_coh-base:+.4f})")
    print(f"    decohered (mixture, avg): negativity = {neg_mix:.4f}  (excess {neg_mix-base:+.4f})")
    print(f"    classical floor:          negativity = {base:.4f}")
    killed = removed > 0.7
    print(f"    -> decoherence removes {removed*100:.0f}% of the genuine negativity"
          f" ({'KILLS it -> ' if killed else ''}witnesses COHERENT (l2) superposition,")
    print(f"       not an l1 classical mixture).")
    out["part2_decoherence"] = {"coherent": neg_coh, "mixture": neg_mix, "floor": base,
                                "frac_removed": removed, "killed": killed}

    # ---- Part 3: negativity vs bundle size k ----
    print(f"\n[3] negativity vs bundle size k (well-separated values):")
    P3 = []
    for k in [1, 2, 3, 4, 6, 8]:
        cs = np.linspace(-4, 14, k)
        negs = []
        for _ in range(5):
            th = make_freqs(N)
            negs.append(negativity(wigner_ville(bundle_profile(cs, th, xs))))
        P3.append(dict(k=k, neg=float(np.mean(negs))))
        print(f"    k={k}: negativity={np.mean(negs):.4f}")
    out["part3_bundle_size"] = P3

    with open("cwf_fpe_wigner_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote cwf_fpe_wigner_results.json")
