# The Physics of Hyperdimensional Computing

**A phase-space lens for Vector Symbolic Architectures (VSA / Hyperdimensional Computing).**

This repository accompanies the manuscript *The Physics of Hyperdimensional Computing:
A Phase-Space Lens for Vector Symbolic Architectures* ([`paper/`](paper/)). It contains
the complete, seeded, CPU-only experiments behind every result, and the figure-generation
script that reproduces every plot in the paper.

VSA/HDC has a mature information-theoretic theory of **capacity** but no **physics** — no
action scale, no temperature, no phase space. Treating an FPE substrate as a *classical
complex wave* (the stance of Computational Wave Field Theory), we supply one, and test what
it buys.

---

## The story in one screen

Fractional Power Encoding (FPE) maps a value `x` to phasors `e^{i x θ}`. We show this is a
genuine **computational phase space** and develop a connected chain of results on it. Each
result carries an explicit honesty fence — the mechanisms are classical coherent-wave /
signal-processing physics (Bochner, Gabor, Wigner–Ville, Wiener–Khinchin), named for VSA.

| # | result | experiment | verdict |
|---|--------|-----------|---------|
| 1 | **FPE = phase space**: kernel = FT(frequencies); resolution × bandwidth = ℏ_c ≈ 1.19 | `cwf_fpe_uncertainty.py`, `cwf_fpe_phasespace.py` | clean (framing) |
| 2 | **Wigner-negativity coherence witness**: bundling is ℓ² (coherent) not ℓ¹ (mixture); decoherence removes 88% | `cwf_fpe_wigner.py` | lands |
| 3 | **Coherence is a usable resource**: interference computes the autocorrelation / a hidden period | `cwf_fpe_interference.py` | lands |
| 4 | **Useful task**: translation-invariant recognition (100% under transposition; naive → chance) | `cwf_fpe_invariant.py` | lands |
| 5 | **New operation**: squeezed / foveated FPE — resolution redistributed at conserved budget | `cwf_fpe_squeezing.py` | new op |
| 6 | **Classical limit**: decoherence + coarse-graining (Husimi); ℏ_c = quantum–classical scale | `cwf_fpe_phasespace.py`, `cwf_fpe_wigner.py` | clean |
| 7 | **Complexity advantage?** — **no algorithmic separation** (classical FFT ties); representational only | `cwf_fpe_complexity.py` | clean negative |
| 8 | **Computational temperature** θ_c for resonator factorization; capacity boundary is a **crossover**, not a transition | `cwf_resonator_halting.py`, `cwf_resonator_scaling.py` | lands / refuted |

### Scope (stated plainly)

This is **classical coherent-wave computation** described in phase-space vocabulary — **not
quantum computation**. VSA cannot realize a quantum speedup: binding is dimension-preserving,
so there is no exponential tensor-product Hilbert space and no entanglement. "Non-classical"
here means *not a classical probability mixture* (a coherent wave), not quantum
non-classicality in the Hudson/contextuality sense. The value of the lens is precisely: a
**unifying framework**, one **new operation** (squeezing), and one **applicable capability**
(invariant recognition) — bounded by two clean negatives.

---

## Layout

```
experiments/   self-contained, seeded NumPy/SciPy experiments + committed *_results.json
paper/         the manuscript (LaTeX + PDF) and make_figures.py (regenerates all figures)
```

## Reproduce

```bash
pip install -r requirements.txt

# any experiment prints its results and (re)writes its *_results.json:
cd experiments
python3 cwf_fpe_wigner.py
python3 cwf_fpe_interference.py
# ... etc.

# regenerate all paper figures (300 dpi) from the committed results:
cd ../paper
python3 make_figures.py
```

Every script is CPU-only, fixed-seed, and toy-scale (seconds to a couple of minutes). The
heavier figure panels (Wigner heatmap, kernels, autocorrelation, resolution profiles) are
recomputed inline with a fixed seed, so the figures are reproducible from a clean checkout.

## The experiments

- **`cwf_fpe_uncertainty.py`** — the kernel is the Fourier transform of the frequency
  distribution; the reciprocal resolution–bandwidth law (the cell ℏ_c); bundling/holding
  capacity ∝ N.
- **`cwf_fpe_phasespace.py`** — the full (x, p) phase space: position/momentum states, the
  conjugate Heisenberg–Weyl flows (translation = binding, boost = modulation), and the ℏ_c → 0
  classical limit (orthogonality + the Husimi coarse-graining).
- **`cwf_fpe_wigner.py`** — the Wigner-negativity witness: cat-state negativity, onset with
  separation, and the decoherence control (−88%).
- **`cwf_fpe_interference.py`** — the coherence as a resource: autocorrelation / hidden-period
  finding by interference, the decoherence + energy controls.
- **`cwf_fpe_invariant.py`** — translation/transposition-invariant recognition (the "chord"
  task), coherent vs naive vs decohered, with a noise-robustness sweep.
- **`cwf_fpe_squeezing.py`** — squeezed / foveated FPE: resolution redistribution, the
  conservation (anti-free-lunch) check, composability.
- **`cwf_fpe_complexity.py`** — the complexity question: bundle readout vs naive O(k²) vs
  classical histogram→FFT; the honest no-separation verdict.
- **`cwf_resonator_halting.py`** — resonator factorization as a halting problem with a noise
  temperature θ_c (the phase diagram).
- **`cwf_resonator_scaling.py`** — finite-size scaling to N = 2048: the capacity boundary is a
  crossover, not a phase transition.

## Citation

If you use this work, please cite the manuscript (see [`paper/`](paper/)) and the
Computational Wave Field Theory book:

> O. Yilmaz. *Computational Wave Field Theory*. Zenodo, v6, 2026.
> [doi:10.5281/zenodo.20665921](https://doi.org/10.5281/zenodo.20665921)

## License

Code released under the MIT License (see [`LICENSE`](LICENSE)). The manuscript text and
figures are © the author.
