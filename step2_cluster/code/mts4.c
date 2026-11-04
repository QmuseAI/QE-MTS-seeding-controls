// mts4: memetic tabu search (MTS) for LABS -- v4 = v3 (mts.c) + optional key=value settings for the extension runs.
// With no key=value arguments the program is bit-for-bit the v3 program: same RNG calls in the same order, same
// output. (Checked on the cluster by the V0 / V2m arms of plan_ext.txt, which must reproduce A0 / A2m exactly.)
//
// usage: mts4 N Eopt mode R S seedfile base_rng [cap] [r_start] [aspiration] [count_init] [kstop] [key=value ...]
//   positional arguments: exactly as in mts.c (see there).
//   key=value settings (all optional):
//     K=<int>        population size (default 100, max KMAX)
//     kcopies=<int>  mode 1 only: members 0..kcopies-1 are copies of the seed, the others are uniform random strings
//                    drawn once per replicate (default K = all copies, i.e. the QE-MTS protocol)
//     nflip=<int>    mode 1 only: every seed copy except member 0 gets nflip distinct random spin flips, drawn once
//                    per replicate (default 0)
//     pmut=<x>       mutation probability = x/N (default 1)
//     tabu=<x>       tabu-search length M is multiplied by x (default 1)
//     solver=<0|1>   0 = MTS (default). 1 = iterated tabu search (ITS), no population: the incumbent (member 0) is
//                    mutated with probability pmut/N per spin, improved by tabu search, and replaced if not worse
//     cfg=<0|1>      1 = append a 5th output column: the configuration that reached Eopt (bit i = 1 <-> s_i = -1),
//                    0 if not found (default 0)
// Random members, flips and the replicate's population use the replicate stream g0, so they are identical for all
// S seeds of a replicate (a replicate = one fixed initial population, as in the paper).
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <math.h>
#define KMAX 400
#define NMAX 64
#define B0 20000LL
#define GROW 1.25

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

// read-only after start -> shared by all threads
static int N, ASP = 1, K = 100, KCOPIES = -1, NFLIP = 0, SOLVER = 0, CFG = 0;
static long Eopt;
static double PMUT = 1.0, TABU = 1.0, PM;          // PM = PMUT / N

typedef struct {
    rng_t g; long long evals; int found; long Eb; uint64_t fcfg;
    int pop[KMAX][NMAX]; long popE[KMAX];
} run_t;

static uint64_t pack(const int *s) { uint64_t x = 0; for (int i = 0; i < N; i++) if (s[i] < 0) x |= 1ULL << i; return x; }

static long energy(const int *s, int *C) {
    long E = 0;
    for (int k = 1; k < N; k++) { int c = 0; for (int i = 0; i + k < N; i++) c += s[i] * s[i + k]; if (C) C[k] = c; E += (long)c * c; }
    return E;
}
static long flipE(const int *s, const int *C, int i) {
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
static long tabu(run_t *R, int *s) {
    rng_t *g = &R->g;
    int p[NMAX], C[NMAX], best[NMAX], tl[NMAX];
    memcpy(p, s, sizeof(int) * N);
    long Ep = energy(p, C); R->evals++; if (Ep <= Eopt) { R->found = 1; return Ep; }
    long Eb = Ep; memcpy(best, p, sizeof(int) * N);
    int M = irand(g, N) + N / 2;
    if (TABU != 1.0) { M = (int)lround(M * TABU); if (M < 1) M = 1; }
    int tmin = M / 10, tex = M / 50; if (tmin < 1) tmin = 1;
    for (int i = 0; i < N; i++) tl[i] = 0;
    for (int t = 1; t <= M; t++) {
        int bi = -1, nties = 0; long bE = 0x7fffffffffffL;
        for (int i = 0; i < N; i++) {
            long e = flipE(p, C, i); R->evals++;
            if (e <= Eopt) { R->found = 1; applyflip(p, C, i); memcpy(s, p, sizeof(int) * N); return e; }
            int admissible = (tl[i] < t) || (ASP && e < Eb);
            if (!admissible) continue;
            if (e < bE) { bE = e; bi = i; nties = 1; }
            else if (e == bE) { nties++; if (irand(g, nties) == 0) bi = i; }
        }
        if (bi < 0) continue;
        applyflip(p, C, bi); Ep = bE;
        tl[bi] = t + tmin + (tex > 0 ? irand(g, tex + 1) : 0);
        if (Ep < Eb) { Eb = Ep; memcpy(best, p, sizeof(int) * N); }
    }
    memcpy(s, best, sizeof(int) * N);
    return Eb;
}
static void advance(run_t *R, long long budget) {
    rng_t *g = &R->g;
    while (!R->found && R->evals < budget) {
        int c[NMAX];
        if (SOLVER == 1) {                                                  // iterated tabu search
            memcpy(c, R->pop[0], sizeof(int) * N);
            for (int i = 0; i < N; i++) if (urand(g) < PM) c[i] = -c[i];
            long Ec = tabu(R, c);
            if (Ec < R->Eb) R->Eb = Ec;
            if (R->found) { R->fcfg = pack(c); break; }
            if (Ec <= R->popE[0]) { memcpy(R->pop[0], c, sizeof(int) * N); R->popE[0] = Ec; }
            continue;
        }
        if (urand(g) < 0.9) {
            int a = irand(g, K), b = irand(g, K); int p1 = R->popE[a] <= R->popE[b] ? a : b;
            a = irand(g, K); b = irand(g, K); int p2 = R->popE[a] <= R->popE[b] ? a : b;
            int cut = 1 + irand(g, N - 1);
            for (int i = 0; i < N; i++) c[i] = i < cut ? R->pop[p1][i] : R->pop[p2][i];
        } else memcpy(c, R->pop[irand(g, K)], sizeof(int) * N);
        for (int i = 0; i < N; i++) if (urand(g) < PM) c[i] = -c[i];
        long Ec = tabu(R, c);
        if (Ec < R->Eb) R->Eb = Ec;
        if (R->found) { R->fcfg = pack(c); break; }
        int v = irand(g, K); memcpy(R->pop[v], c, sizeof(int) * N); R->popE[v] = Ec;
    }
}
static int cmpll(const void *a, const void *b) {
    long long x = *(const long long *)a, y = *(const long long *)b; return (x > y) - (x < y);
}
static int opt(const char *arg, const char *key, double *v) {
    size_t n = strlen(key);
    if (strncmp(arg, key, n) == 0 && arg[n] == '=') { *v = atof(arg + n + 1); return 1; }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 8) { fprintf(stderr, "usage: mts4 N Eopt mode R S seedfile base_rng [cap] [r_start] [aspiration] [count_init] [kstop] [key=value ...]\n"); return 2; }
    // split positional and key=value arguments
    char *pos[16]; int npos = 0;
    for (int i = 1; i < argc; i++) {
        double v;
        if (strchr(argv[i], '=') == NULL) { if (npos < 16) pos[npos++] = argv[i]; continue; }
        if (opt(argv[i], "K", &v)) K = (int)v;
        else if (opt(argv[i], "kcopies", &v)) KCOPIES = (int)v;
        else if (opt(argv[i], "nflip", &v)) NFLIP = (int)v;
        else if (opt(argv[i], "pmut", &v)) PMUT = v;
        else if (opt(argv[i], "tabu", &v)) TABU = v;
        else if (opt(argv[i], "solver", &v)) SOLVER = (int)v;
        else if (opt(argv[i], "cfg", &v)) CFG = (int)v;
        else { fprintf(stderr, "unknown setting %s\n", argv[i]); return 2; }
    }
    if (npos < 7) { fprintf(stderr, "need 7 positional arguments\n"); return 2; }
    N = atoi(pos[0]); Eopt = atol(pos[1]); int mode = atoi(pos[2]);
    int R = atoi(pos[3]), S = atoi(pos[4]); const char *sf = pos[5]; uint64_t base = strtoull(pos[6], 0, 10);
    long long cap = npos > 7 ? atoll(pos[7]) : 1000000000000LL;
    int r0 = npos > 8 ? atoi(pos[8]) : 0;
    ASP = npos > 9 ? atoi(pos[9]) : 1;
    int count_init = npos > 10 ? atoi(pos[10]) : (mode == 0);
    int kstop = npos > 11 ? atoi(pos[11]) : S;
    if (N < 3 || N > NMAX) { fprintf(stderr, "N must be in [3,%d]\n", NMAX); return 2; }
    if (mode < 0 || mode > 2) { fprintf(stderr, "mode must be 0, 1 or 2\n"); return 2; }
    if (K < 1 || K > KMAX) { fprintf(stderr, "K must be in [1,%d]\n", KMAX); return 2; }
    if (KCOPIES < 0 || KCOPIES > K) KCOPIES = K;
    if (KCOPIES < 1) KCOPIES = 1;
    if (NFLIP < 0 || NFLIP > N) { fprintf(stderr, "nflip must be in [0,N]\n"); return 2; }
    if ((KCOPIES < K || NFLIP > 0) && mode != 1) { fprintf(stderr, "kcopies/nflip need mode 1\n"); return 2; }
    if (SOLVER == 1 && mode == 2) { fprintf(stderr, "solver=1 needs mode 0 or 1\n"); return 2; }
    if (S < 1 || cap < 1) { fprintf(stderr, "need S >= 1 and cap >= 1\n"); return 2; }
    if (kstop <= 0 || kstop > S) kstop = S;
    if (kstop < S / 2 + 1) { fprintf(stderr, "kstop=%d < S/2+1=%d would make the replicate median inexact\n", kstop, S / 2 + 1); return 2; }
    PM = PMUT / N;                                  // PMUT = 1 -> identical to v3's 1.0 / N

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
    static int init[KMAX][NMAX];
    run_t *runs = malloc((size_t)S * sizeof(run_t));
    long long *T = malloc((size_t)S * sizeof(long long));
    if (!runs || !T) { fprintf(stderr, "out of memory\n"); return 1; }
    for (int r = r0; r < r0 + R; r++) {
        rng_t g0; seed_rng(&g0, base, (uint64_t)r, 0xFFFFFFFFULL);
        for (int m = 0; m < K; m++) {
            uint64_t x = 0;
            if (mode == 1) x = seeds[r];
            else if (mode == 2) x = seeds[(long)r * K + m];
            if (mode == 0 || (mode == 1 && m >= KCOPIES)) {               // uniform random member
                for (int i = 0; i < N; i++) init[m][i] = (rnd(&g0) & 1) ? 1 : -1;
                continue;
            }
            for (int i = 0; i < N; i++) init[m][i] = ((x >> i) & 1) ? -1 : 1;
            if (mode == 1 && NFLIP > 0 && m > 0) {                        // perturbed seed copy
                int idx[NMAX]; for (int i = 0; i < N; i++) idx[i] = i;
                for (int j = 0; j < NFLIP; j++) {
                    int q = j + irand(&g0, N - j); int t = idx[j]; idx[j] = idx[q]; idx[q] = t;
                    init[m][idx[j]] = -init[m][idx[j]];
                }
            }
        }
        for (int sd = 0; sd < S; sd++) {
            run_t *Q = &runs[sd];
            seed_rng(&Q->g, base + 1000003ULL, (uint64_t)r, (uint64_t)sd);
            Q->evals = 0; Q->found = 0; Q->Eb = 0x7fffffffffffL; Q->fcfg = 0;
            for (int m = 0; m < K; m++) {
                memcpy(Q->pop[m], init[m], sizeof(int) * N); Q->popE[m] = energy(Q->pop[m], 0);
                if (count_init) Q->evals++;
                if (Q->popE[m] < Q->Eb) Q->Eb = Q->popE[m];
                if (Q->popE[m] <= Eopt && !Q->found) { Q->found = 1; Q->fcfg = pack(Q->pop[m]); }
            }
            if (SOLVER == 1) {                                              // ITS starts from the best member
                int bm = 0; for (int m = 1; m < K; m++) if (Q->popE[m] < Q->popE[bm]) bm = m;
                if (bm) { memcpy(Q->pop[0], Q->pop[bm], sizeof(int) * N); Q->popE[0] = Q->popE[bm]; }
            }
        }
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
        for (int sd = 0; sd < S; sd++) {
            printf("%d %d %lld %ld", r, sd, runs[sd].found ? runs[sd].evals : -(runs[sd].evals > 0 ? runs[sd].evals : 1), runs[sd].Eb);
            if (CFG) printf(" %" PRIu64, runs[sd].found ? runs[sd].fcfg : (uint64_t)0);
            printf("\n");
        }
        fflush(stdout);
    }
    free(runs); free(T); free(seeds);
    return 0;
}
