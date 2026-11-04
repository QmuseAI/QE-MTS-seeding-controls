"""Check the optimal-energy table used by the runs:
  1. every run-length-encoded optimum (Packebusch & Mertens 2016) has length N and energy EOPT[N];
  2. its 8 symmetry images have the same energy (checks the symmetry code used for oracle seeds A7*);
  3. writes eopt_all.txt (N Eopt) for mts.c / make_tasks.py.
Optional extra check (slow, compile mts first):  python verify_eopt.py --mts ./mts --runs 8
  runs short MTS searches with target Eopt-4 (must never succeed) as an independent sanity check.
"""
import sys, subprocess
from labs_common import EOPT, RLE, rle_to_spins, energy_spins, symmetric_variants

bad = 0
for N, code in sorted(RLE.items()):
    s = rle_to_spins(code); E = energy_spins(s)
    ok = (len(s) == N) and (E == EOPT[N]) and all(energy_spins(v) == E for v in symmetric_variants(s))
    print(f"N={N:2d} len={len(s):2d} E={E:4d} table={EOPT[N]:4d} {'OK' if ok else 'MISMATCH'}")
    bad += not ok
with open("eopt_all.txt", "w") as f:
    for N in sorted(EOPT): f.write(f"{N} {EOPT[N]}\n")
print("wrote eopt_all.txt;", "ALL OK" if bad == 0 else f"{bad} MISMATCHES")

if "--mts" in sys.argv:
    mts = sys.argv[sys.argv.index("--mts") + 1]
    runs = int(sys.argv[sys.argv.index("--runs") + 1]) if "--runs" in sys.argv else 4
    for N in sorted(RLE):
        # target below the optimum can never be reached: every run must end capped (-1) with best >= EOPT
        out = subprocess.run([mts, str(N), str(EOPT[N] - 4), "0", "1", str(runs), "none", "77", "20000000"],
                             capture_output=True, text=True).stdout.split("\n")
        best = min(int(l.split()[3]) for l in out if l.strip())
        print(f"N={N}: best found in {runs} capped runs = {best} (table {EOPT[N]}) {'OK' if best >= EOPT[N] else 'BELOW TABLE?!'}")
sys.exit(1 if bad else 0)
