#!/usr/bin/env python3
"""HIGH-STATISTICS theta_c sweep for the resonator (n = 500 instances per point).

Same construction as cwf_resonator_halting.py Part 2 (N=256, D=128, F=2, bipolar,
T=150), but 10x the instances and a finer temperature grid, so the optimum carries
error bars (binomial SE ~ 0.022 at p=0.5).  Writes
cwf_resonator_thetahi_results.json; the committed part2_theta_c (n=50) is untouched.
"""
import importlib.util
import json

import numpy as np

spec = importlib.util.spec_from_file_location("rh", "cwf_resonator_halting.py")
rh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rh)

N, D, F, T = 256, 128, 2, 150
NINST = 500
THETAS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


def main():
    rows = []
    print(f"N={N}, D={D}, F={F}, T={T}, n={NINST}/point")
    for th in THETAS:
        succ, nohalt, spur = rh.prob(N, D, F, "bipolar", T, th, NINST, 90000)
        se = float(np.sqrt(succ * (1 - succ) / NINST))
        rows.append(dict(theta=th, succ=float(succ), nohalt=float(nohalt),
                         se=se))
        print(f"  theta_c={th:>5}: success={succ:.3f} +/- {se:.3f}  "
              f"non-halting={nohalt:.3f}")
    out = dict(params=dict(N=N, D=D, F=F, T=T, n_inst=NINST), rows=rows)
    with open("cwf_resonator_thetahi_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote cwf_resonator_thetahi_results.json")


if __name__ == "__main__":
    main()
