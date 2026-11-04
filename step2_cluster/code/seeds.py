"""Generate the initial populations (seed files) for the seeded MTS arms.

usage: python seeds.py --arms A1 A2 A5 A6 A7a A7b --nmin 20 --nmax 50 --R 100 --outdir ../seeds [--nproc 64]

Arms (K = 100, replicate r uses line r, or lines r*K..r*K+K-1 for diverse populations):
  A1  : one uniformly random string, replicated K times            (zero-information control)
  A2  : lowest-energy string among 1e5 uniform samples, replicated K times
        (the paper's QE-MTS protocol with the quantum sampler replaced by uniform sampling; same shot count)
  A5  : the K lowest-energy DISTINCT strings among 1e5 uniform samples   (same budget, diverse population)
  A6  : the K lowest-energy DISTINCT strings among 1e7 uniform samples   (larger budget, cf. paper App. D)
  A7a : oracle seed - a random symmetry image of a known optimum with round(N/4) random spins flipped, x K
  A7b : oracle seed - same with round(N/8) spins flipped, x K            (seed-quality sweep far beyond A2)
Seed-quality sweep of the replicated protocol (added 2026-10-22; all x K, i.e. the QE-MTS protocol):
  A2k : lowest-energy string among 1e3 uniform samples
  A2m : lowest-energy string among 1e6 uniform samples
  A3  : lowest-energy string among 1e7 uniform samples = the best member of the A6 population of the same
        replicate (same RNG stream as A6) -> A3 vs A6 is "replicated vs diverse" with identical information
  A8  : best NON-optimal string visited by a tabu walk of 1000*N single-flip moves from a random start
        (states with E = Eopt are forbidden), i.e. a low-energy seed that is not the answer
Ties in energy are broken uniformly at random. Every (arm, N) uses its own fixed RNG stream.
Also writes seed_stats.csv: per (arm, N, r) seed energy and Hamming distance to the nearest image of the
known optimum (for diverse arms: best member and population mean energy).
"""
import argparse, os, sys
import numpy as np
from multiprocessing import Pool
from labs_common import EOPT, RLE, rle_to_spins, spins_to_int, energy_ints, popcount64, symmetric_variants

K = 100
ARM_ID = {"A1": 1, "A2": 2, "A5": 5, "A6": 6, "A7a": 71, "A7b": 72, "A2k": 21, "A2m": 22, "A3": 6, "A8": 8}
TABU_MOVES = 1000        # A8: tabu walk length in units of N single-flip moves


def uniform(rng, N, n):
    return rng.integers(0, 1 << N, n, dtype=np.uint64)


def top_k_distinct(rng, x, E, k):
    u, idx = np.unique(x, return_index=True)
    Eu = E[idx]
    perm = rng.permutation(len(u)); u, Eu = u[perm], Eu[perm]          # random tie breaking
    o = np.argsort(Eu, kind="stable")[:k]
    return u[o], Eu[o]


def opt_images(N):
    return np.array([spins_to_int(v) for v in symmetric_variants(rle_to_spins(RLE[N]))], dtype=np.uint64)


def ham_to_opt(x, N):
    imgs = opt_images(N)
    return popcount64(np.asarray(x, dtype=np.uint64)[:, None] ^ imgs[None, :]).min(1)


def best_of_uniform(rng, N, n, chunk=1000000):
    """lowest-energy string among n uniform samples (chunked), random tie breaking"""
    bx, bE = None, None
    for i in range(0, n, chunk):
        xs = uniform(rng, N, min(chunk, n - i)); E = energy_ints(xs, N)
        m = E.min(); cand = xs[E == m]
        if bE is None or m < bE: bx, bE = [cand], m
        elif m == bE: bx.append(cand)
    c = np.concatenate(bx)
    return c[rng.integers(len(c)):][:1]


def tabu_walk_best(rng, N, eopt, moves):
    """tabu walk of `moves` single-flip steps from a random string; states with E = eopt are forbidden.
    Returns the best visited string (as N-bit int) and its energy (> eopt)."""
    s = np.where(rng.integers(0, 2, N) == 0, 1, -1).astype(np.int64)
    k = np.arange(1, N); j = np.arange(N)
    jp, jm = j[:, None] + k[None, :], j[:, None] - k[None, :]
    vp, vm = jp < N, jm >= 0
    jp, jm = np.where(vp, jp, 0), np.where(vm, jm, 0)
    C = np.array([int(np.dot(s[:N - kk], s[kk:])) for kk in k], dtype=np.int64)
    E = int((C * C).sum())
    tabu = np.zeros(N, np.int64); bestE, best = None, None
    if E > eopt: bestE, best = E, s.copy()
    for t in range(1, moves + 1):
        P = np.where(vp, s[jp], 0) + np.where(vm, s[jm], 0)
        Cn = C[None, :] - 2 * s[:, None] * P
        En = (Cn * Cn).sum(1)
        ok = (En > eopt) & ((tabu < t) | ((bestE is not None) & (En < (bestE if bestE is not None else 0))))
        if not ok.any(): ok = En > eopt
        cand = np.flatnonzero(ok & (En == En[ok].min()))
        f = int(cand[rng.integers(len(cand))])
        s[f] = -s[f]; C = Cn[f]; E = int(En[f])
        tabu[f] = t + max(1, N // 10) + int(rng.integers(0, max(1, N // 50) + 1))
        if bestE is None or E < bestE: bestE, best = E, s.copy()
    return np.array([spins_to_int(best)], dtype=np.uint64), bestE


def make(job):
    arm, N, R, outdir = job
    rng = np.random.default_rng([ARM_ID[arm], N, 20261020])
    lines, stats = [], []
    for r in range(R):
        if arm == "A1":
            x = uniform(rng, N, 1); pop = x
        elif arm == "A2":
            xs = uniform(rng, N, 100000); E = energy_ints(xs, N); i = int(np.argmin(E)); x = xs[i:i + 1]; pop = x
        elif arm == "A5":
            xs = uniform(rng, N, 100000); pop, _ = top_k_distinct(rng, xs, energy_ints(xs, N), K)
        elif arm in ("A6", "A3"):                                         # A3 = best member of the A6 population
            cx, cE = [], []
            for _ in range(10):                                            # 10 chunks of 1e6 = 1e7 samples
                xs = uniform(rng, N, 1000000); E = energy_ints(xs, N)
                j = np.argpartition(E, 8 * K)[:8 * K]; cx.append(xs[j]); cE.append(E[j])
            pop, _ = top_k_distinct(rng, np.concatenate(cx), np.concatenate(cE), K)
            if arm == "A3": pop = pop[:1]
        elif arm == "A2k":
            pop = best_of_uniform(rng, N, 1000)
        elif arm == "A2m":
            pop = best_of_uniform(rng, N, 1000000)
        elif arm == "A8":
            pop, _ = tabu_walk_best(rng, N, EOPT[N], TABU_MOVES * N)
        elif arm in ("A7a", "A7b"):
            d = int(round(N / 4)) if arm == "A7a" else int(round(N / 8))
            imgs = opt_images(N); x0 = int(imgs[rng.integers(len(imgs))])
            for i in rng.choice(N, d, replace=False): x0 ^= (1 << int(i))
            x = np.array([x0], dtype=np.uint64); pop = x
        else:
            raise ValueError(arm)
        if len(pop) < (K if arm in ("A5", "A6") else 1):
            raise RuntimeError(f"{arm} N={N} r={r}: only {len(pop)} distinct seeds")
        Ep = energy_ints(pop, N); hp = ham_to_opt(pop, N)
        lines += [str(int(v)) for v in pop]
        stats.append(f"{arm},{N},{r},{int(Ep.min())},{float(Ep.mean()):.2f},{int(hp[np.argmin(Ep)])},{int(hp.min())},{len(pop)}")
    with open(os.path.join(outdir, f"seeds_{arm}_N{N}.txt"), "w") as f: f.write("\n".join(lines) + "\n")
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["A1", "A2", "A5", "A6", "A7a", "A7b"])
    ap.add_argument("--nmin", type=int, default=20); ap.add_argument("--nmax", type=int, default=50)
    ap.add_argument("--nlist", type=str, default="", help="optional per-arm N ranges, e.g. 'A6:27-37,A7a:27-37'")
    ap.add_argument("--R", type=int, default=100); ap.add_argument("--outdir", default="../seeds")
    ap.add_argument("--nproc", type=int, default=os.cpu_count())
    ap.add_argument("--stats", default="seed_stats.csv", help="name of the statistics file written in --outdir")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    rng_over = {}
    for item in filter(None, a.nlist.split(",")):
        arm, rr = item.split(":"); lo, hi = map(int, rr.split("-")); rng_over[arm] = (lo, hi)
    jobs = []
    for arm in a.arms:
        lo, hi = rng_over.get(arm, (a.nmin, a.nmax))
        for N in range(lo, hi + 1):
            if arm.startswith("A7") and N not in RLE: continue
            jobs.append((arm, N, a.R, a.outdir))
    jobs.sort(key=lambda j: (j[0] not in ("A6", "A3", "A2m"), -j[1]))                            # heaviest first
    with Pool(min(a.nproc, len(jobs))) as p:
        res = p.map(make, jobs, chunksize=1)
    with open(os.path.join(a.outdir, a.stats), "w") as f:
        f.write("arm,N,r,seedE_min,seedE_mean,ham_bestseed_to_opt,ham_min_to_opt,n_distinct\n")
        for s in res: f.write("\n".join(s) + "\n")
    print(f"wrote {len(jobs)} seed files to {a.outdir}")
