// Memetic tabu search (MTS) for LABS, following Gomez Cadavid et al., QMI 8:92 (2026), Alg. 1-3
// (K=100, p_comb=0.9, p_mut=1/N, tournament size 2, offspring replaces a uniformly random member),
// tabu search as in Gallardo, Cotta & Fernandez (2009): M = U{0..N-1} + floor(N/2) iterations,
// tenure = t + M/10 + U{0..M/50}.
//
// TTS = number of objective evaluations until an energy <= Eopt is first evaluated. One evaluation =
// one call that yields the energy of one configuration (a full energy() or one incremental flipE()).
//
// usage: mts N Eopt mode R S seedfile base_rng [cap_evals] [r_start] [aspiration] [count_init] [kstop]
//   mode 0 : random initial population (K uniform random strings), drawn once per replicate
//   mode 1 : replicate r uses the single string on line r of seedfile, copied K times      (paper's QE-MTS protocol)
//   mode 2 : replicate r uses the K strings on lines r*K .. r*K+K-1 of seedfile (diverse population)
//   cap_evals  : abort a run after this many evaluations. default 1e12 (no cap in practice)
//   r_start    : run replicates r_start .. r_start+R-1 (job splitting). default 0
//   aspiration : 1 = tabu move allowed if it improves on the best of this tabu run (Gallardo 2009, paper App. F text)
//                0 = no aspiration (paper Alg. 2 pseudocode as printed). default 1
//   count_init : 1 = count the K energy evaluations of the initial population (mode 0 always evaluates them;
//                with 1 they are also counted for seeded modes). default 0 for modes 1/2, 1 for mode 0.
//                (the paper does not count the cost of producing the quantum seeds)
//   kstop      : early stop of a replicate (v3). The S runs of a replicate are advanced together in rising
//                evaluation budgets; as soon as the kstop smallest TTS values are known EXACTLY, the other runs
//                are stopped and reported as right-censored. default S (= no early stop, old behaviour).
//                kstop must be >= S/2+1 so that the replicate median is exact; 0.7*S also keeps the seed-level
//                bootstrap median exact (P(bootstrap median > T_(kstop)) ~ 1e-5 for S=100).
//   Eopt = 0   : never "found"; just search and report the best energy (column 4) -> for optimum checks.
// seedfile: one unsigned 64-bit integer per line; bit i = 0 -> s_i = +1.
// output: one line per (r, s): "r s evals best_energy". evals < 0 means censored (not found): the run was
//   stopped after |evals| evaluations, either by the cap or by the early stop. (analysis: < 0 -> +inf)
// RNG: xoshiro256** seeded through splitmix64 from (base_rng, replicate, seed index) -> independent streams.
// Every run is a deterministic function of (base_rng, r, s): pausing between generations and resuming does
// not change its trajectory, so TTS values are identical to the sequential v2 code, for any thread count.
// Threads: compile with -fopenmp; the runs of one replicate are spread over OMP_NUM_THREADS threads.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#define K 100
#define NMAX 64
#define B0 20000LL          // first evaluation budget of the staged schedule
#define GROW 1.25           // budget growth per stage

typedef struct { uint64_t s[4]; } rng_t;
static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
static inline uint64_t rnd(rng_t *g) {
    uint64_t *S4 = g->s;
    uint64_t r = rotl(S4[1] * 5, 7) * 9, t = S4[1] << 17;
    S4[2] ^= S4[0]; S4[3] ^= S4[1]; S4[1] ^= S4[2]; S4[0] ^= S4[3]; S4[2] ^= t; S4[3] = rotl(S4[3], 45);
    return r;
}
static uint64_t splitmix(uint64_t *x) {
    uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL; return z ^ (z >> 31);
}
static void seed_rng(rng_t *g, uint64_t a, uint64_t b, uint64_t c) {
    uint64_t x = a * 0x9E3779B97F4A7C15ULL;
    uint64_t h = splitmix(&x) ^ (b + 1) * 0xD1B54A32D192ED03ULL; x = h;
    h = splitmix(&x) ^ (c + 1) * 0x8CB92BA72F3D8DD7ULL; x = h;
    for (int i = 0; i < 4; i++) g->s[i] = splitmix(&x);
}
static inline double urand(rng_t *g) { return (rnd(g) >> 11) * (1.0 / 9007199254740992.0); }
static inline int irand(rng_t *g, int n) { return (int)(urand(g) * n); }

static int N, ASP = 1; static long Eopt;          // read-only after start -> shared by all threads

typedef struct {                                  // complete state of one MTS run (one seed)
    rng_t g; long long evals; int found; long Eb;
    int pop[K][NMAX]; long popE[K];
} run_t;

static long energy(const int *s, int *C) {
    long E = 0;
    for (int k = 1; k < N; k++) { int c = 0; for (int i = 0; i + k < N; i++) c += s[i] * s[i + k]; if (C) C[k] = c; E += (long)c * c; }
    return E;
}
static long flipE(const int *s, const int *C, int i) {     // energy after flipping s_i, O(N)
    long E = 0;
    for (int k = 1; k < N; k++) {
        int d = 0; if (i + k < N) d += s[i + k]; if (i - k >= 0) d += s[i - k];
        int c = C[k] - 2 * s[i] * d; E += (long)c * c;
    }
    return E;
}
static void applyflip(int *s, int *C, int i) {
    for (int k = 1; k < N; k++) { int d = 0; if (i + k < N) d += s[i + k]; if (i - k >= 0) d += s[i - k]; C[k] -= 2 * s[i] * d; }
    s[i] = -s[i];
}
// tabu search from s; on return s holds the best configuration of this tabu run; returns its energy
static long tabu(run_t *R, int *s) {
    rng_t *g = &R->g;
    int p[NMAX], C[NMAX], best[NMAX], tl[NMAX];
    memcpy(p, s, sizeof(int) * N);
    long Ep = energy(p, C); R->evals++; if (Ep <= Eopt) { R->found = 1; return Ep; }
    long Eb = Ep; memcpy(best, p, sizeof(int) * N);
    int M = irand(g, N) + N / 2; int tmin = M / 10, tex = M / 50; if (tmin < 1) tmin = 1;
    for (int i = 0; i < N; i++) tl[i] = 0;
    for (int t = 1; t <= M; t++) {
        int bi = -1, nties = 0; long bE = 0x7fffffffffffL;
        for (int i = 0; i < N; i++) {
            long e = flipE(p, C, i); R->evals++;
            if (e <= Eopt) { R->found = 1; applyflip(p, C, i); memcpy(s, p, sizeof(int) * N); return e; }
            int admissible = (tl[i] < t) || (ASP && e < Eb);
            if (!admissible) continue;
            if (e < bE) { bE = e; bi = i; nties = 1; }
            else if (e == bE) { nties++; if (irand(g, nties) == 0) bi = i; }   // uniform tie breaking
        }
        if (bi < 0) continue;                                              // every move tabu: wait
        applyflip(p, C, bi); Ep = bE;
        tl[bi] = t + tmin + (tex > 0 ? irand(g, tex + 1) : 0);
        if (Ep < Eb) { Eb = Ep; memcpy(best, p, sizeof(int) * N); }
    }
    memcpy(s, best, sizeof(int) * N);
    return Eb;
}
// run MTS generations until found or evals >= budget. Same loop body and RNG order as v2's
// "while (!found && evals < cap)", so a run paused here and resumed later follows the identical trajectory.
static void advance(run_t *R, long long budget) {
    rng_t *g = &R->g;
    while (!R->found && R->evals < budget) {
        int c[NMAX];
        if (urand(g) < 0.9) {
            int a = irand(g, K), b = irand(g, K); int p1 = R->popE[a] <= R->popE[b] ? a : b;
            a = irand(g, K); b = irand(g, K); int p2 = R->popE[a] <= R->popE[b] ? a : b;
            int cut = 1 + irand(g, N - 1);
            for (int i = 0; i < N; i++) c[i] = i < cut ? R->pop[p1][i] : R->pop[p2][i];
        } else memcpy(c, R->pop[irand(g, K)], sizeof(int) * N);
        for (int i = 0; i < N; i++) if (urand(g) < 1.0 / N) c[i] = -c[i];
        long Ec = tabu(R, c);
        if (Ec < R->Eb) R->Eb = Ec;
        if (R->found) break;
        int v = irand(g, K); memcpy(R->pop[v], c, sizeof(int) * N); R->popE[v] = Ec;
    }
}
static int cmpll(const void *a, const void *b) {
    long long x = *(const long long *)a, y = *(const long long *)b; return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    if (argc < 8) { fprintf(stderr, "usage: mts N Eopt mode R S seedfile base_rng [cap] [r_start] [aspiration] [count_init] [kstop]\n"); return 2; }
    N = atoi(argv[1]); Eopt = atol(argv[2]); int mode = atoi(argv[3]);
    int R = atoi(argv[4]), S = atoi(argv[5]); const char *sf = argv[6]; uint64_t base = strtoull(argv[7], 0, 10);
    long long cap = argc > 8 ? atoll(argv[8]) : 1000000000000LL;
    int r0 = argc > 9 ? atoi(argv[9]) : 0;
    ASP = argc > 10 ? atoi(argv[10]) : 1;
    int count_init = argc > 11 ? atoi(argv[11]) : (mode == 0);
    int kstop = argc > 12 ? atoi(argv[12]) : S;
    if (N < 3 || N > NMAX) { fprintf(stderr, "N must be in [3,%d]\n", NMAX); return 2; }
    if (mode < 0 || mode > 2) { fprintf(stderr, "mode must be 0, 1 or 2\n"); return 2; }
    if (S < 1 || cap < 1) { fprintf(stderr, "need S >= 1 and cap >= 1\n"); return 2; }
    if (kstop <= 0 || kstop > S) kstop = S;
    if (kstop < S / 2 + 1) { fprintf(stderr, "kstop=%d < S/2+1=%d would make the replicate median inexact\n", kstop, S / 2 + 1); return 2; }

    uint64_t *seeds = NULL; long nseeds = 0;
    if (mode > 0) {
        FILE *f = fopen(sf, "r");
        if (!f) { fprintf(stderr, "cannot open seed file %s\n", sf); return 1; }
        long cap_s = 1024; seeds = malloc(cap_s * sizeof(uint64_t)); uint64_t v;
        while (fscanf(f, "%" SCNu64, &v) == 1) {
            if (nseeds == cap_s) { cap_s *= 2; seeds = realloc(seeds, cap_s * sizeof(uint64_t)); }
            seeds[nseeds++] = v;
        }
        fclose(f);
        long need = (long)(r0 + R) * (mode == 2 ? K : 1);
        if (nseeds < need) { fprintf(stderr, "seed file %s has %ld entries, need %ld\n", sf, nseeds, need); return 1; }
    }
    static int init[K][NMAX];
    run_t *runs = malloc((size_t)S * sizeof(run_t));
    long long *T = malloc((size_t)S * sizeof(long long));
    if (!runs || !T) { fprintf(stderr, "out of memory\n"); return 1; }
    for (int r = r0; r < r0 + R; r++) {
        // initial population of this replicate (identical for all its S seeds, as in the paper)
        rng_t g0; seed_rng(&g0, base, (uint64_t)r, 0xFFFFFFFFULL);
        for (int m = 0; m < K; m++) {
            uint64_t x = 0;
            if (mode == 1) x = seeds[r];
            else if (mode == 2) x = seeds[(long)r * K + m];
            for (int i = 0; i < N; i++)
                init[m][i] = (mode == 0) ? ((rnd(&g0) & 1) ? 1 : -1) : (((x >> i) & 1) ? -1 : 1);
        }
        for (int sd = 0; sd < S; sd++) {
            run_t *Q = &runs[sd];
            seed_rng(&Q->g, base + 1000003ULL, (uint64_t)r, (uint64_t)sd);
            Q->evals = 0; Q->found = 0; Q->Eb = 0x7fffffffffffL;
            for (int m = 0; m < K; m++) {
                memcpy(Q->pop[m], init[m], sizeof(int) * N); Q->popE[m] = energy(Q->pop[m], 0);
                if (count_init) Q->evals++;
                if (Q->popE[m] < Q->Eb) Q->Eb = Q->popE[m];
            }
            if (Q->Eb <= Eopt) Q->found = 1;
        }
        // staged schedule: all unfinished runs advance to a common budget; once kstop runs have finished,
        // the unfinished ones only need to pass T* = kstop-th smallest finished TTS (then T > T* is certain)
        long long B = B0;
        for (;;) {
            int nf = 0;
            for (int sd = 0; sd < S; sd++) if (runs[sd].found) T[nf++] = runs[sd].evals;
            long long target;
            if (nf >= kstop) {
                qsort(T, nf, sizeof(long long), cmpll);
                target = T[kstop - 1] < cap ? T[kstop - 1] : cap;
                int pending = 0;
                for (int sd = 0; sd < S; sd++) if (!runs[sd].found && runs[sd].evals < target) pending = 1;
                if (!pending) break;
            } else {
                int alive = 0;
                for (int sd = 0; sd < S; sd++) if (!runs[sd].found && runs[sd].evals < cap) alive = 1;
                if (!alive) break;
                target = B < cap ? B : cap;
                B = (long long)(B * GROW) + 1;
            }
            #pragma omp parallel for schedule(dynamic, 1)
            for (int sd = 0; sd < S; sd++)
                if (!runs[sd].found && runs[sd].evals < target) advance(&runs[sd], target);
        }
        for (int sd = 0; sd < S; sd++)
            printf("%d %d %lld %ld\n", r, sd, runs[sd].found ? runs[sd].evals : -(runs[sd].evals > 0 ? runs[sd].evals : 1), runs[sd].Eb);
        fflush(stdout);
    }
    free(runs); free(T); free(seeds);
    return 0;
}
