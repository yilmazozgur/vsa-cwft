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
genuine **computational phase space** and develop a connected chain of results on it — from
the uncertainty cell through a coherence witness and a new operation to a proven
order-optimality theorem for the bundle-as-sketch, a hardware bit-precision rule, and the
identity that RoPE positional encoding *is* FPE. Each result carries an explicit honesty
fence — the mechanisms are classical coherent-wave / signal-processing physics (Bochner,
Gabor, Wigner–Ville, Wiener–Khinchin, Alon–Matias–Szegedy), named for VSA.

| # | result | experiment | verdict |
|---|--------|-----------|---------|
| 1 | **FPE = phase space**: kernel = FT(frequencies); resolution × bandwidth = ℏ_c ≈ 1.19 | `cwf_fpe_uncertainty.py`, `cwf_fpe_phasespace.py` | clean (framing) |
| 2 | **Wigner-negativity coherence witness**: bundling is ℓ² (coherent) not ℓ¹ (mixture); decoherence removes 88% | `cwf_fpe_wigner.py` | lands |
| 3 | **Coherence is a usable resource**: interference computes the autocorrelation / a hidden period | `cwf_fpe_interference.py` | lands |
| 4 | **Useful task**: translation-invariant recognition (100% under transposition; naive → chance) | `cwf_fpe_invariant.py` | lands |
| 5 | **New operation**: squeezed / foveated FPE — resolution redistributed at conserved budget | `cwf_fpe_squeezing.py`, `cwf_fpe_fovea_task.py` | new op |
| 6 | **Classical limit**: decoherence + coarse-graining (Husimi); ℏ_c = quantum–classical scale | `cwf_fpe_phasespace.py`, `cwf_fpe_wigner.py` | clean |
| 7 | **Complexity advantage?** — **no algorithmic separation** (classical FFT ties); representational only | `cwf_fpe_complexity.py` | clean negative |
| 8 | **The bundle is an order-optimal relational sketch**: Var = Θ(k²/N) empirical characteristic function; matches the AGMS ±1 sketch and Woodruff's Ω(1/ε²) lower bound (theorem + numerics) | `cwf_fpe_ams.py` | theorem |
| 9 | **Computational temperature** θ_c for resonator factorization; capacity boundary is a **crossover**, not a transition | `cwf_resonator_halting.py`, `cwf_resonator_scaling.py` | lands / refuted |
| 10 | **The cell on hardware**: coherence survives quantization only above ≥ 2 phase bits (photonic jitter / PCM bit-depth) | `cwf_fpe_hardware.py`, `cwf_fpe_quantize.py` | lands |
| 11 | **Connections**: RoPE *is* FPE (an elementary identity); a linear-attention / SSM state is a running relational sketch | `cwf_fpe_rope.py`, `cwf_fpe_ssm.py` | bridge |

### Scope (stated plainly)

This is **classical coherent-wave computation** described in phase-space vocabulary — **not
quantum computation**. VSA cannot realize a quantum speedup: binding is dimension-preserving,
so there is no exponential tensor-product Hilbert space and no entanglement. "Non-classical"
here means *not a classical probability mixture* (a coherent wave), not quantum
non-classicality in the Hudson/contextuality sense. The value of the lens is precisely: a
**unifying framework**, one **new operation** (squeezing), one **applicable capability**
(invariant recognition), and one **proven optimality result** (the bundle is an order-optimal
relational sketch) — bounded by two clean negatives.

The same dimension-preserving structure sets a sharp reach limit into nonclassical
statistics: a hypervector space can *host* single-system (KCBS-type) contextuality but not
the tensor-product (Bell-type) tier, and whether binding can *generate* rather than merely
host that geometry is left open. That thread — the Computational Wave Field Theory
tie-in — is developed in a separate foundations manuscript; this repository is the
mainstream phase-space paper and its experiments.

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
- **`cwf_fpe_squeezing.py`**, **`cwf_fpe_fovea_task.py`** — squeezed / foveated FPE:
  resolution redistribution, the conservation (anti-free-lunch) check, composability, and a
  downstream task where the redistributed resolution pays off.
- **`cwf_fpe_complexity.py`** — the complexity question: bundle readout vs naive O(k²) vs
  classical histogram→FFT; the honest no-separation verdict.
- **`cwf_fpe_ams.py`** — the bundle as an Alon–Matias–Szegedy / empirical-characteristic-
  function sketch: the Var = Θ(k²/N) error law, the order-optimality match against the AGMS
  ±1 sketch, and the Woodruff Ω(1/ε²) lower bound.
- **`cwf_fpe_hardware.py`**, **`cwf_fpe_quantize.py`** — the cell under finite hardware:
  phase quantization / jitter, and the ≥ 2-phase-bit floor for coherence to survive.
- **`cwf_fpe_rope.py`**, **`cwf_fpe_ssm.py`** — the connections: RoPE positional encoding
  as an instance of FPE, and a linear-attention / state-space-model state as a running
  relational sketch.
- **`cwf_resonator_halting.py`**, **`cwf_resonator_scaling.py`**,
  **`cwf_resonator_tcap.py`**, **`cwf_resonator_thetahi.py`** — resonator factorization as a
  halting problem with a noise temperature θ_c: the phase diagram, finite-size scaling to
  N = 2048 (the capacity boundary is a crossover, not a phase transition), and the
  temperature/capacity sweeps.
- Supporting checks: **`cwf_fpe_stats.py`**, **`cwf_fpe_variance.py`**,
  **`cwf_fpe_weyl.py`**, **`cwf_fpe_highdim.py`**, **`cwf_fpe_localcell.py`**,
  **`cwf_fpe_models.py`**, **`cwf_fpe_beyond_capacity.py`** — the statistical, variance,
  Weyl-relation, high-dimensional, local-cell, cross-model, and beyond-capacity controls
  behind the numbers in the paper.

## Citation

If you use this work, please cite the manuscript (see [`paper/`](paper/)) and the
Computational Wave Field Theory book:

> O. Yilmaz. *Computational Wave Field Theory*. Zenodo, v6, 2026.
> [doi:10.5281/zenodo.20665921](https://doi.org/10.5281/zenodo.20665921)

## License

Code released under the MIT License (see [`LICENSE`](LICENSE)). The manuscript text and
figures are © the author.
