# The Physics of Hyperdimensional Computing: code and results

This repository holds the code and the results behind the paper

> O. Yilmaz, *The Physics of Hyperdimensional Computing: A Phase-Space Lens for Vector Symbolic
> Architectures* (2026).

Every experiment is seeded and runs on a CPU. Every data figure in the paper regenerates from the
result files committed here.

## The idea in brief

Vector Symbolic Architectures (VSA), also called Hyperdimensional Computing (HDC), store continuous
values with Fractional Power Encoding (FPE). FPE writes a value `x` into the phases `exp(i x θ_j)` of
a long complex vector. The paper treats such a vector as a classical wave and shows that FPE is a
computational phase space. Value and frequency are conjugate, and resolution times bandwidth is a
fixed cell. Bundling adds amplitudes coherently, and excess Wigner negativity measures that
coherence. The coherence also computes. One power-spectrum readout returns the histogram of all
pairwise differences of the stored values, and it gives translation-invariant recognition without
decoding. Read as a budget, the uncertainty relation gives squeezing, a warp that moves resolution
to where it is needed.

Everything here is classical computation with coherent waves. The paper makes no quantum claim.

## Layout

```
experiments/      31 seeded experiment scripts, each with its committed *_results.json
make_figures.py   regenerates every figure of the paper, schematics included, into figures/
requirements.txt  minimum versions of the dependencies
```

## Requirements

Python 3.10 or later, with

```
numpy>=2.0
scipy>=1.10
matplotlib>=3.6
threadpoolctl>=3.0
```

The committed results were produced with Python 3.10.12, NumPy 2.2.6, SciPy 1.15.3,
Matplotlib 3.10.3 and threadpoolctl 3.6.0.

## Reproduce

```bash
pip install -r requirements.txt

# every figure, from the committed result files (about 20 seconds)
python3 make_figures.py

# one experiment: it prints its results and rewrites its *_results.json
cd experiments
python3 cwf_fpe_wigner.py

# every experiment, one after another (most scripts take seconds to a few minutes)
for s in cwf_*.py; do python3 "$s"; done
```

Run the experiment scripts from inside `experiments/`, because a few of them load a sibling script
by its file name. `make_figures.py` finds `experiments/` on its own. To point it at another folder,
set the environment variable `CWFT_EXPERIMENTS`.

## Where each result comes from

Section, figure and table numbers refer to the paper and its Supplemental Material (S).

| Paper | Result | Script |
|---|---|---|
| Sec. 4, Figs. 2 and S5A | the kernel is the Fourier transform of the codebook, the cell ℏ_c, the capacity check | `cwf_fpe_uncertainty.py` |
| Sec. 4 | the Weyl phase of translation and boost | `cwf_fpe_weyl.py` |
| Sec. 4, Fig. 3 | which of the two capacity counts binds | `cwf_fpe_budgets.py` |
| Sec. 4, Supp. S2 | the cell on other VSA carriers | `cwf_fpe_models.py` |
| Sec. 5, Fig. 4 | the Wigner-negativity witness | `cwf_fpe_wigner.py` |
| Sec. 5, Fig. S19A | removal of the excess negativity over codebook seeds | `cwf_fpe_stats.py` |
| Sec. 5, Supp. S3 | phase quantization | `cwf_fpe_quantize.py` |
| Supp. S3 | sign conventions and their unit tests | `cwf_fpe_wigner_convention.py` |
| Sec. 6, Fig. S9 | the relational readout and hidden-period finding | `cwf_fpe_interference.py` |
| Sec. 7, Figs. 5 and S11 | translation-invariant recognition and its collisions | `cwf_fpe_invariant.py` |
| Sec. 7 | the baseline on raw values | `cwf_fpe_invariant_baseline.py` |
| Sec. 7, Supp. S4, Fig. S13 | recognition in m dimensions | `cwf_fpe_invariant_md.py` |
| Sec. 8, Fig. 6A | squeezed (foveated) FPE | `cwf_fpe_squeezing.py` |
| Sec. 8, Fig. 6B | the local cell | `cwf_fpe_localcell.py` |
| Sec. 8, Supp. S5, Table S2, Fig. S14 | the decoding task under three budgets | `cwf_fpe_fovea_task.py` |
| Supp. S6.1, Fig. S16 | velocity templates for values in motion | `cwf_fpe_chirp.py` |
| Supp. S6.2, Fig. S17 | the joint (x, p) code for trajectories | `cwf_fpe_trajectory.py` |
| Supp. S6.2 | regime recognition | `cwf_fpe_regime.py` |
| Supp. S7, Fig. S19B and C | orthogonal codes and coarse-graining | `cwf_fpe_phasespace.py` |

Figures S8 and S20 are computed inside `make_figures.py`, with fixed seeds.

The remaining scripts come from earlier, longer versions of the paper: `cwf_fpe_ams.py`,
`cwf_fpe_beyond_capacity.py`, `cwf_fpe_bispectrum.py`, `cwf_fpe_complexity.py`,
`cwf_fpe_hardware.py`, `cwf_fpe_hashjoin.py`, `cwf_fpe_highdim.py`, `cwf_fpe_variance.py`,
`cwf_rope_fovea.py`, `cwf_resonator_halting.py`, `cwf_resonator_scaling.py` and
`cwf_resonator_thetahi.py`. The current paper doesn't use them. They stay here because
`make_figures.py` still draws their figures.

## Expected runtimes

Measured one script at a time on an Intel Core i9-14900K, while other work ran on the machine.

| Script | Time |
|---|---|
| `make_figures.py` | 16 s |
| `cwf_fpe_ams.py` | 17 s |
| `cwf_fpe_beyond_capacity.py` | 5.3 min |
| `cwf_fpe_bispectrum.py` | 19 s |
| `cwf_fpe_budgets.py` | 41 s |
| `cwf_fpe_chirp.py` | 42 s |
| `cwf_fpe_complexity.py` | 1.5 min |
| `cwf_fpe_fovea_task.py` | 3.0 min |
| `cwf_fpe_hardware.py` | 4.2 min |
| `cwf_fpe_hashjoin.py` | 48 s |

The remaining scripts are being timed, and this table will be completed.

## Citation

If you use this code, please cite the paper:

> O. Yilmaz, *The Physics of Hyperdimensional Computing: A Phase-Space Lens for Vector Symbolic
> Architectures* (2026).

The phase-space stance comes from Computational Wave Field Theory:

> O. Yilmaz, *Computational Wave Field Theory*, Zenodo, v9 (2026).
> [doi:10.5281/zenodo.22997207](https://doi.org/10.5281/zenodo.22997207)

## License

The code and the result files are released under the MIT License (see [`LICENSE`](LICENSE)).
