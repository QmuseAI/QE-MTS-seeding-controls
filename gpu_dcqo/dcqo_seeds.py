"""DCQO seeds for the QE-MTS protocol (impulse-regime DCQO with the first-order counterdiabatic term), CUDA-Q.

Physics (Gomez Cadavid et al., QMI 8:92, Sec. 3.1 and App. B), written out:
  H_ad(l) = (1-l) H_i + l H_f,  H_i = sum_i h X_i with h = -1 (ground state |+>^N),  H_f = E(s) - const (their Eq. 2,
  built here directly from E(s) = sum_k C_k^2, so the two-body indices are i, i+2k).
  First-order nested commutator: A = i a1(l) O1, O1 = [H_ad, d_l H_ad] = [H_i, H_f]   (independent of l),
  a1(l) = -G1/G2(l), G1 = Tr[O1^+ O1], G2(l) = Tr[O2^+ O2], O2 = [H_ad, O1].
  Impulse regime: H(t) = l'(t) A(l(t)). All H(t) are proportional to the same operator O1, so the evolution is exactly
      U = exp(Theta * O1) = exp(-i Theta G),  G = i[H_i, H_f] = sum_A J_A sum_{i in A} 2h Y_i Z_{A minus i},
      Theta = int_0^1 a1(l) dl,
  independent of the schedule l(t) and of T. "n_trot -> infinity" in the paper is this exact exponential; here it is
  Trotterised with n_steps steps (check convergence with --converge).
  G1, G2(l) are computed exactly with Pauli-string algebra (no 2^N matrices); --selftest checks them, the sign
  convention of cudaq.exp_pauli and the whole circuit against dense matrices at small N.

Seeds (same formats as code/seeds.py):
  per replicate r: 1e5 shots (--shots); Q1 = lowest-energy shot (ties at random) -> seeds_Q1_N<N>.txt (R lines),
  Q5 = 100 lowest-energy distinct shots -> seeds_Q5_N<N>.txt (R*100 lines); statistics -> seed_stats_dcqo.csv
  per N -> seed_stats_dcqo_N<N>.csv and dcqo_summary_N<N>.json; --merge writes seed_stats_dcqo.csv.
  Shots of one N come from one simulation (the state does not change between replicates); they are shuffled and
  split into R groups, which is the same distribution as R independent runs of 1e5 shots.

usage (one GPU per N up to N=33 in single precision; N=34-36 need --mgpu with mpirun over the 8 GPUs):
  python dcqo_seeds.py --selftest                                   # dense check, N=6..10   (minutes)
  python dcqo_seeds.py --converge 27 --steps 1 2 4 8 16 32          # Trotter convergence    (minutes)
  python dcqo_seeds.py --estimate 27 30 33                          # projected GPU time per N
  CUDA_VISIBLE_DEVICES=0 python dcqo_seeds.py --N 27 --steps 16 --outdir seeds_dcqo     # production, one N
  mpirun -np 8 python dcqo_seeds.py --N 35 --steps 16 --mgpu --outdir seeds_dcqo
Requires: cudaq (CUDA-Q), numpy, scipy; labs_common.py in the same folder (standalone package gpu_dcqo/).
"""
import argparse, os, sys, time, json
import numpy as np
from scipy import integrate, linalg
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "code")); sys.path.insert(0, _here)   # labs_common.py next to this file first
from labs_common import EOPT, RLE, rle_to_spins, spins_to_int, energy_ints, symmetric_variants

H_X = -1.0
pc = np.bitwise_count if hasattr(np, "bitwise_count") else None
if pc is None:
    from labs_common import popcount64 as pc


# ---------------------------------------------------------------- Hamiltonians as Pauli sums (x mask, z mask, coefficient)
def hf_terms(N):
    """Z-strings of E(s) = sum_k (sum_i s_i s_{i+k})^2 without the constant: {z mask: coefficient}"""
    T = {}
    for k in range(1, N):
        for i in range(N - k):
            for j in range(N - k):
                if i == j: continue
                m = (1 << i) ^ (1 << (i + k)) ^ (1 << j) ^ (1 << (j + k))
                T[m] = T.get(m, 0) + 1
    return {m: c for m, c in T.items() if c}


def as_arrays(d):
    k = list(d)
    return (np.array([x for x, z in k], np.uint64), np.array([z for x, z in k], np.uint64),
            np.array([d[q] for q in k], complex))


def hamiltonians(N):
    hf = hf_terms(N)
    Hf = {(0, m): float(c) for m, c in hf.items()}
    Hi = {(1 << i, 0): H_X for i in range(N)}
    G = {}
    for m, J in hf.items():
        for i in range(N):
            if m >> i & 1:
                G[(1 << i, m)] = G.get((1 << i, m), 0.0) + 2.0 * H_X * J
    return Hi, Hf, G


def reduce_terms(x, z, c):
    if len(x) == 0: return x, z, c
    key = np.ascontiguousarray(np.stack([x, z], 1)).view([("x", "<u8"), ("z", "<u8")]).ravel()
    u, inv = np.unique(key, return_inverse=True)
    cs = np.bincount(inv, weights=c.real, minlength=len(u)) + 1j * np.bincount(inv, weights=c.imag, minlength=len(u))
    keep = np.abs(cs) > 1e-12
    return u["x"][keep].astype(np.uint64), u["z"][keep].astype(np.uint64), cs[keep]


def commutator(A, B, chunk_pairs=2e7):
    """[A, B] for Pauli sums A, B given as (x, z, c) arrays; P(x,z) = i^{|x&z|} X^x Z^z"""
    xa, za, ca = A; xb, zb, cb = B
    out_x, out_z, out_c = [], [], []
    step = max(1, int(chunk_pairs // max(1, len(xb))))
    ab = pc(xb & zb).astype(np.int64)
    for s0 in range(0, len(xa), step):
        X1 = xa[s0:s0 + step, None]; Z1 = za[s0:s0 + step, None]; C1 = ca[s0:s0 + step, None]
        anti = ((pc(X1 & zb[None, :]) + pc(Z1 & xb[None, :])) & 1).astype(bool)
        i1, i2 = np.nonzero(anti)
        if len(i1) == 0: continue
        x1, z1, x2, z2 = X1[i1, 0], Z1[i1, 0], xb[i2], zb[i2]
        x3, z3 = x1 ^ x2, z1 ^ z2
        e = (pc(x1 & z1).astype(np.int64) + ab[i2] + 2 * pc(z1 & x2).astype(np.int64) - pc(x3 & z3).astype(np.int64)) % 4
        coef = 2.0 * (1j ** e) * C1[i1, 0] * cb[i2]
        x3, z3, coef = reduce_terms(x3, z3, coef)
        out_x.append(x3); out_z.append(z3); out_c.append(coef)
    if not out_x: return np.zeros(0, np.uint64), np.zeros(0, np.uint64), np.zeros(0, complex)
    return reduce_terms(np.concatenate(out_x), np.concatenate(out_z), np.concatenate(out_c))


def inner(A, B):
    """Re <A, B> = Re sum conj(a_k) b_k  (Hilbert-Schmidt / 2^N)"""
    xa, za, ca = A; xb, zb, cb = B
    ka = np.ascontiguousarray(np.stack([xa, za], 1)).view([("x", "<u8"), ("z", "<u8")]).ravel()
    kb = np.ascontiguousarray(np.stack([xb, zb], 1)).view([("x", "<u8"), ("z", "<u8")]).ravel()
    _, ia, ib = np.intersect1d(ka, kb, assume_unique=True, return_indices=True)
    return float(np.real(np.sum(np.conj(ca[ia]) * cb[ib])))


def theta_of(N, verbose=True):
    t0 = time.time()
    Hi, Hf, G = hamiltonians(N)
    Ga = as_arrays(G); Hia = as_arrays(Hi); Hfa = as_arrays(Hf)
    Xp = commutator(Hia, Ga); Yp = commutator(Hfa, Ga)
    g1 = float(np.sum(np.abs(Ga[2]) ** 2))                                # ||O1||^2 = ||G||^2
    nx, ny, ixy = float(np.sum(np.abs(Xp[2]) ** 2)), float(np.sum(np.abs(Yp[2]) ** 2)), inner(Xp, Yp)
    g2 = lambda l: (1 - l) ** 2 * nx + 2 * l * (1 - l) * ixy + l ** 2 * ny
    a1 = lambda l: -g1 / g2(l)
    Theta = integrate.quad(a1, 0, 1, limit=200)[0]
    info = dict(N=N, n_terms_G=len(G), n_terms_Hf=len(Hf), G1=g1, G2_0=nx, G2_1=ny, G2_cross=ixy, Theta=Theta,
                seconds=round(time.time() - t0, 1))
    if verbose: print(f"N={N}: {len(G)} Pauli terms in G, Theta = {Theta:.6g}  ({info['seconds']} s)")
    return G, Theta, info, g2, g1


def words_thetas(G, Theta, N, nsteps, sign):
    words, th = [], []
    for (x, z), g in G.items():
        i = int(x).bit_length() - 1
        w = ["I"] * N
        for q in range(N):
            if (z >> q) & 1: w[q] = "Z"
        w[i] = "Y"
        words.append("".join(w)); th.append(sign * Theta * g / nsteps)
    return words, th


# ---------------------------------------------------------------- CUDA-Q
def setup_cudaq(mgpu, fp64=False):
    import cudaq
    opt = ("mgpu," if mgpu else "") + ("fp64" if fp64 else "fp32")
    cudaq.set_target("nvidia", option=opt)
    if mgpu: cudaq.mpi.initialize()
    return cudaq


def make_kernel(cudaq):
    @cudaq.kernel
    def dcqo(n: int, thetas: list[float], words: list[cudaq.pauli_word], nsteps: int):
        q = cudaq.qvector(n)
        h(q)
        for s in range(nsteps):
            for j in range(len(thetas)):
                exp_pauli(thetas[j], q, words[j])
    return dcqo


def sample_ints(cudaq, kernel, N, words, th, nsteps, shots, batch=None):
    """sample shots; returns uint64 array, bit q = 1 <-> qubit q measured in |1> <-> s_q = -1.
    Every cudaq.sample call re-simulates the circuit, so all shots are taken in ONE call by default
    (batch=None); a batch size only trades simulation time for host memory."""
    vals, cnts = [], []
    left = shots
    while left > 0:
        b = left if batch is None else min(batch, left); left -= b
        res = cudaq.sample(kernel, N, th, [cudaq.pauli_word(w) for w in words], nsteps, shots_count=b)
        for bits, c in res.items():
            vals.append(int(bits[::-1], 2)); cnts.append(c)                     # char q = qubit q
    vals = np.array(vals, np.uint64); cnts = np.array(cnts, np.int64)
    # merge identical bitstrings from different batches
    u, inv = np.unique(vals, return_inverse=True)
    return u, np.bincount(inv, weights=cnts).astype(np.int64)


# ---------------------------------------------------------------- dense reference (small N)
def dense_check(N, cudaq=None, kernel=None, nsteps=64):
    dim = 1 << N; idx = np.arange(dim, dtype=np.uint64)
    E = energy_ints(idx, N).astype(float)
    Hf = np.diag(E)
    Hi = np.zeros((dim, dim))
    for i in range(N):
        Hi[np.arange(dim), np.arange(dim) ^ (1 << i)] += H_X
    O1 = Hi @ Hf - Hf @ Hi
    G, Theta, info, g2, g1 = theta_of(N, verbose=False)
    G1d = np.trace(O1.T @ O1) / dim
    ok = True
    for l in (0.0, 0.3, 0.7, 1.0):
        Had = (1 - l) * Hi + l * Hf; O2 = Had @ O1 - O1 @ Had
        G2d = np.trace(O2.conj().T @ O2).real / dim
        rel = abs(G2d - g2(l)) / G2d
        ok &= rel < 1e-9
        print(f"   N={N} l={l:.1f}: G2 dense {G2d:.6g}  Pauli {g2(l):.6g}  rel.diff {rel:.1e}")
    print(f"   N={N}: G1 dense {G1d:.6g}  Pauli {g1:.6g}")
    ok &= abs(G1d - g1) / G1d < 1e-9
    psi = linalg.expm(Theta * O1) @ (np.ones(dim) / np.sqrt(dim))
    levels = np.unique(E); pE_exact = np.array([np.sum(np.abs(psi[E == e]) ** 2) for e in levels])
    print(f"   N={N}: exact DCQO: mean energy {np.sum(np.abs(psi)**2 * E):.4f} (uniform {E.mean():.4f}), "
          f"P(optimum) {pE_exact[0]:.4f} (uniform {np.mean(E == E.min()):.4f})")
    best = None
    if cudaq is not None:
        for sign in (-1.0, +1.0):
            words, th = words_thetas(G, Theta, N, nsteps, sign)
            st = np.array(cudaq.get_state(kernel, N, th, [cudaq.pauli_word(w) for w in words], nsteps))
            p = np.abs(st) ** 2
            for order in ("little", "big"):
                if order == "little": bits = idx
                else: bits = np.array([int(format(int(b), f"0{N}b")[::-1], 2) for b in idx], np.uint64)
                Eo = energy_ints(bits, N)
                pE = np.array([np.sum(p[Eo == e]) for e in levels])
                tv = 0.5 * np.sum(np.abs(pE - pE_exact))
                print(f"   N={N} cudaq sign {sign:+.0f} ({order}-endian index): total variation to exact P(E) = {tv:.2e}")
                if best is None or tv < best[0]: best = (tv, sign)
        print(f"   -> use --sign {best[1]:+.0f}" + ("" if best[0] < 1e-3 else "   WARNING: no convention matches"))
    return ok, (best[1] if best else None)


# ---------------------------------------------------------------- seeds
def opt_images(N):
    return np.array([spins_to_int(v) for v in symmetric_variants(rle_to_spins(RLE[N]))], dtype=np.uint64)


def ham_to_opt(x, N):
    imgs = opt_images(N)
    return pc(np.asarray(x, np.uint64)[:, None] ^ imgs[None, :]).min(1)


def make_seeds(N, u, cnt, R, shots, K, rng, outdir, info):
    allshots = np.repeat(u, cnt); rng.shuffle(allshots)
    if len(allshots) < R * shots: raise SystemExit(f"only {len(allshots)} shots for N={N}")
    q1, q5, stats = [], [], []
    for r in range(R):
        x = allshots[r * shots:(r + 1) * shots]; E = energy_ints(x, N)
        m = E.min(); cand = x[E == m]; b = cand[rng.integers(len(cand))]
        q1.append(int(b))
        ux, ix = np.unique(x, return_index=True); Eu = E[ix]
        perm = rng.permutation(len(ux)); ux, Eu = ux[perm], Eu[perm]
        o = np.argsort(Eu, kind="stable")[:K]; pop = ux[o]
        if len(pop) < K: raise SystemExit(f"N={N} r={r}: only {len(pop)} distinct shots")
        q5 += [int(v) for v in pop]
        hb = int(ham_to_opt(np.array([b], np.uint64), N)[0]); hp = ham_to_opt(pop, N)
        stats.append(f"Q1,{N},{r},{int(m)},{float(m):.2f},{hb},{hb},1")
        stats.append(f"Q5,{N},{r},{int(Eu[o].min())},{float(Eu[o].mean()):.2f},{int(hp[0])},{int(hp.min())},{len(pop)}")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"seeds_Q1_N{N}.txt"), "w") as f: f.write("\n".join(map(str, q1)) + "\n")
    with open(os.path.join(outdir, f"seeds_Q5_N{N}.txt"), "w") as f: f.write("\n".join(map(str, q5)) + "\n")
    with open(os.path.join(outdir, f"seed_stats_dcqo_N{N}.csv"), "w") as f:     # one file per N (parallel runs);
        f.write("arm,N,r,seedE_min,seedE_mean,ham_bestseed_to_opt,ham_min_to_opt,n_distinct\n")   # merged by --merge
        f.write("\n".join(stats) + "\n")
    Eall = energy_ints(u, N)
    info.update(mean_energy_over_Eopt=float(np.sum(Eall * cnt) / cnt.sum() / EOPT[N]),
                P_opt=float(cnt[Eall == EOPT[N]].sum() / cnt.sum()),
                Q1_seed_E_over_Eopt=float(np.mean([energy_ints(np.array([v], np.uint64), N)[0] for v in q1]) / EOPT[N]))
    with open(os.path.join(outdir, f"dcqo_summary_N{N}.json"), "w") as f: f.write(json.dumps(info) + "\n")
    print(f"N={N}: seeds written; mean shot energy {info['mean_energy_over_Eopt']:.3f} Eopt, P(opt) {info['P_opt']:.2e}, "
          f"Q1 seed energy {info['Q1_seed_E_over_Eopt']:.3f} Eopt")


# ---------------------------------------------------------------- main
ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, nargs="*", default=[])
ap.add_argument("--steps", type=int, nargs="+", default=[16], help="Trotter steps (production: one value)")
ap.add_argument("--sign", type=float, default=-1.0, help="theta = sign*Theta*g/n (exp_pauli = exp(i theta P) -> -1)")
ap.add_argument("--R", type=int, default=100); ap.add_argument("--shots", type=int, default=100000)
ap.add_argument("--K", type=int, default=100); ap.add_argument("--outdir", default=os.path.join(_here, "seeds_dcqo"))
ap.add_argument("--mgpu", action="store_true"); ap.add_argument("--fp64", action="store_true")
ap.add_argument("--selftest", action="store_true"); ap.add_argument("--theta-only", action="store_true")
ap.add_argument("--converge", type=int, default=0, help="N for the Trotter convergence check")
ap.add_argument("--estimate", type=int, nargs="*", default=[], help="N values: projected GPU time")
ap.add_argument("--seed", type=int, default=20261023)
ap.add_argument("--merge", action="store_true", help="merge the per-N statistics files in --outdir and exit")
a = ap.parse_args()

if a.merge:
    import glob
    fs = sorted(glob.glob(os.path.join(a.outdir, "seed_stats_dcqo_N*.csv")), key=lambda f: int(f.split("_N")[-1][:-4]))
    with open(os.path.join(a.outdir, "seed_stats_dcqo.csv"), "w") as out:
        out.write("arm,N,r,seedE_min,seedE_mean,ham_bestseed_to_opt,ham_min_to_opt,n_distinct\n")
        for f in fs: out.writelines(open(f).readlines()[1:])
    with open(os.path.join(a.outdir, "dcqo_summary.txt"), "w") as out:
        for f in sorted(glob.glob(os.path.join(a.outdir, "dcqo_summary_N*.json"))): out.write(open(f).read())
    print(f"merged {len(fs)} sizes into {a.outdir}/seed_stats_dcqo.csv and dcqo_summary.txt")
    sys.exit(0)

if a.theta_only:
    for N in a.N: theta_of(N)
    sys.exit(0)

cudaq = setup_cudaq(a.mgpu, a.fp64); kernel = make_kernel(cudaq)
rank0 = (not a.mgpu) or cudaq.mpi.rank() == 0

if a.selftest:
    allok = True
    for N in (6, 8, 10):
        ok, sign = dense_check(N, cudaq, kernel); allok &= ok
    print("SELFTEST", "OK" if allok else "FAILED (Pauli algebra vs dense)")
    sys.exit(0)

if a.converge:
    N = a.converge; G, Theta, info, _, _ = theta_of(N)
    ref = None
    for n in sorted(a.steps):
        words, th = words_thetas(G, Theta, N, n, a.sign)
        t0 = time.time(); u, c = sample_ints(cudaq, kernel, N, words, th, n, 200000); dt = time.time() - t0
        E = energy_ints(u, N); mE = np.sum(E * c) / c.sum() / EOPT[N]
        q = np.sort(np.repeat(E, c))[: max(1, c.sum() // 1000)].mean() / EOPT[N]
        if rank0: print(f"N={N} steps={n:3d}: mean shot energy {mE:.4f} Eopt, mean of best 0.1% {q:.4f} Eopt  ({dt:.0f} s)")
    if rank0: print("choose the smallest number of steps after which both numbers stop changing (to ~0.5%)")
    sys.exit(0)

if a.estimate:
    for N in a.estimate:
        G, Theta, info, _, _ = theta_of(N, verbose=False)
        words, th = words_thetas(G, Theta, N, 1, a.sign)
        nt = min(300, len(words))
        sample_ints(cudaq, kernel, N, words[:2], th[:2], 1, 10)              # warm-up (JIT, allocation)
        t0 = time.time(); sample_ints(cudaq, kernel, N, words[:nt], th[:nt], 1, 10); t1 = time.time() - t0
        t0 = time.time(); sample_ints(cudaq, kernel, N, words[:nt // 3], th[:nt // 3], 1, 10); t2 = time.time() - t0
        per = max(1e-6, (t1 - t2) / (nt - nt // 3))
        if rank0: print(f"N={N}: {len(words)} terms, ~{per*1e3:.1f} ms per term -> {per*len(words)/60:.1f} min per Trotter step "
                        f"(x number of steps; plus a few seconds of sampling)")
    sys.exit(0)

rng = np.random.default_rng(a.seed)
for N in a.N:
    G, Theta, info, _, _ = theta_of(N, verbose=rank0)
    n = a.steps[0]; words, th = words_thetas(G, Theta, N, n, a.sign)
    t0 = time.time(); u, c = sample_ints(cudaq, kernel, N, words, th, n, a.R * a.shots); info["sim_seconds"] = round(time.time() - t0)
    info.update(steps=n, sign=a.sign, shots_per_replicate=a.shots, R=a.R)
    if rank0: make_seeds(N, u, c, a.R, a.shots, a.K, np.random.default_rng([a.seed, N]), a.outdir, info)
