"""Validate the per-replicate early stop (mts v3, kfrac = 0.70) against runs without early stop (kfrac = 1).

usage: python check_earlystop.py --run RUN [--pairs A0f:A0 A2f:A2] [--nmin 27 --nmax 33 --R 20]
A0f / A2f use the same seeds and RNG streams as A0 / A2, so each run (r, s) must follow the same trajectory:
  - a run finished in both files must have exactly the same TTS;
  - a run that was right-censored after E evaluations by the early stop must have TTS > E in the full run;
  - the replicate median (over seeds) must be identical.
Prints one line per (pair, N) and an overall verdict.
"""
import argparse, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--pairs", nargs="+", default=["A0f:A0", "A2f:A2"])
ap.add_argument("--nmin", type=int, default=27); ap.add_argument("--nmax", type=int, default=33)
ap.add_argument("--R", type=int, default=20)
a = ap.parse_args()

def load(arm, N, r):
    p = os.path.join(a.run, "parts", f"{arm}_N{N}_r{r}.txt")
    if not os.path.exists(p): return None
    d = np.loadtxt(p, ndmin=2)
    return {int(s): int(e) for _, s, e, _ in d} if d.size else None

bad_total, checked_total = 0, 0
for pair in a.pairs:
    full, ref = pair.split(":")
    for N in range(a.nmin, a.nmax + 1):
        n_same = n_cens_ok = n_bad = n_rep = n_med_ok = n_missing = 0
        for r in range(a.R):
            F, E = load(full, N, r), load(ref, N, r)
            if F is None or E is None: n_missing += 1; continue
            n_rep += 1
            for s, ef in F.items():
                ee = E.get(s)
                if ee is None: n_bad += 1; continue
                if ee > 0:                                        # finished in the early-stopped run
                    if ef == ee: n_same += 1
                    else: n_bad += 1
                else:                                             # censored after |ee| evaluations
                    if ef < 0 or ef > -ee: n_cens_ok += 1          # still running at the stop (or capped later)
                    else: n_bad += 1
            mf = np.median([v if v > 0 else np.inf for v in F.values()])
            me = np.median([v if v > 0 else np.inf for v in E.values()])
            n_med_ok += (mf == me)
        bad_total += n_bad; checked_total += n_same + n_cens_ok
        print(f"{full} vs {ref} N={N}: replicates {n_rep} (missing {n_missing}); identical TTS {n_same}; "
              f"censored runs consistent {n_cens_ok}; mismatches {n_bad}; identical replicate medians {n_med_ok}/{n_rep}")
print(f"OVERALL: {checked_total} runs checked, {bad_total} mismatches -> "
      f"{'early stop is exact' if bad_total == 0 and checked_total > 0 else 'CHECK FAILED or no data'}")
