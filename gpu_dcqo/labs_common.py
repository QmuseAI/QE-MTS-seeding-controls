"""Shared LABS utilities: energies, optimal energies and one optimal sequence per N.

Optimal energies and run-length-encoded optimal sequences are transcribed from
T. Packebusch and S. Mertens, J. Phys. A 49, 165001 (2016), arXiv:1512.02475, Tables 1-2
(exhaustive branch & bound, exact for N <= 66). Run-length code: digits = lengths of runs of
equal spins, a=10, b=11, c=12. verify_eopt.py re-computes every energy from the sequence.
"""
import numpy as np

EOPT = {14: 19, 15: 15, 16: 24, 17: 32, 18: 25, 19: 29, 20: 26, 21: 26, 22: 39, 23: 47, 24: 36, 25: 36,
        26: 45, 27: 37, 28: 50, 29: 62, 30: 59, 31: 67, 32: 64, 33: 64, 34: 65, 35: 73, 36: 82, 37: 86,
        38: 87, 39: 99, 40: 108, 41: 108, 42: 101, 43: 109, 44: 122, 45: 118, 46: 131, 47: 135,
        48: 140, 49: 136, 50: 153}

RLE = {20: "5113112321", 21: "27221111121", 22: "51221111233", 23: "212121111632", 24: "2236111112121",
       25: "337111121221", 26: "21212111116322", 27: "34313131211211", 28: "34313131211212",
       29: "212112131313431", 30: "551212111113231", 31: "7332212211112111", 32: "71112111133221221",
       33: "742112111111122221", 34: "842112111111122221", 35: "7122122111121111332",
       36: "3632311131212111211", 37: "844211211111122221", 38: "8442112111111122221",
       39: "82121121234321111111", 40: "44412112131121313131", 41: "343111111222281211211",
       42: "313131341343112112112", 43: "1132432111117212112213", 44: "525313113111222111211121",
       45: "82121121231234321111111", 46: "823431231211212211111111", 47: "923431231211212211111111",
       48: "3111111832143212221121121", 49: "215131311224112241141141", 50: "215131311224112241141142"}


def rle_to_spins(code):
    """run-length code -> array of +-1 spins, first run = +1"""
    runs = [int(c, 36) for c in code]
    s, sign = [], 1
    for L in runs:
        s += [sign] * L; sign = -sign
    return np.array(s, dtype=np.int64)


def energy_spins(s):
    s = np.asarray(s, dtype=np.int64); N = len(s)
    return int(sum(int(np.dot(s[:N - k], s[k:])) ** 2 for k in range(1, N)))


def spins_to_int(s):
    """bit i = 0 <-> s_i = +1 (convention of mts.c)"""
    x = 0
    for i, v in enumerate(s):
        if v < 0: x |= (1 << i)
    return x


def popcount64(x):
    """vectorised popcount for uint64 arrays (uses np.bitwise_count when available, NumPy >= 2.0)"""
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x).astype(np.int64)
    x = x.astype(np.uint64)
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int64)


def energy_ints(x, N):
    """LABS energies of an array of N-bit integers (bit i = 0 <-> s_i = +1)"""
    x = np.asarray(x, dtype=np.uint64); E = np.zeros(x.shape, np.int64)
    for k in range(1, N):
        m = np.uint64((1 << (N - k)) - 1)
        c = (N - k) - 2 * popcount64((x ^ (x >> np.uint64(k))) & m)
        E += c * c
    return E


def symmetric_variants(s):
    """the 8 symmetry images (complement, reversal, alternate complement) of a spin sequence"""
    s = np.asarray(s); alt = np.array([(-1) ** i for i in range(len(s))])
    out = []
    for a in (s, s[::-1]):
        for b in (a, a * alt):
            out += [b, -b]
    return out
