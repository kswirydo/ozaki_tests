#!/usr/bin/env python3
"""
Quick test comparing Original vs Fixed Ozaki II implementations
"""
import numpy as np

def extended_gcd(a, b):
    if a == 0: return b, 0, 1
    gcd, x1, y1 = extended_gcd(b % a, a)
    return gcd, y1 - (b // a) * x1, x1

def modinv(a, m):
    a = a % m
    if a == 0: return 0
    gcd, x, _ = extended_gcd(a, m)
    return x % m

def symmetric_mod(x, p):
    r = x % p
    if isinstance(r, np.ndarray):
        r[r > p // 2] = r[r > p // 2] - p
    elif r > p // 2:
        r = r - p
    return r

def ozaki2(A, B, num_mod, fixed=False):
    """Ozaki Scheme II implementation"""
    if fixed:
        moduli_all = [256, 255, 253, 251, 247, 241, 239, 233, 229, 227,
                      223, 217, 211, 199, 197, 193, 191, 181, 179, 173]
    else:
        moduli_all = [256, 255, 253, 251, 247, 239, 233, 229, 
                      227, 223, 217, 211, 199, 197, 193, 191]
    moduli = moduli_all[:num_mod]
    
    M = 1
    for m in moduli: M *= m
    
    crt_weights = []
    for i, pi in enumerate(moduli):
        Pi = M // pi
        qi = modinv(Pi % pi, pi)
        crt_weights.append(qi * Pi)
    
    amax = np.max(np.abs(A), axis=1)
    bmax = np.max(np.abs(B), axis=0)
    
    if fixed:
        log2P_half = np.log2(float(M - 1)) / 2.0
        sftA = (np.floor(log2P_half - 1) - np.floor(np.log2(amax + 1e-300))).astype(int)
        sftB = (np.floor(log2P_half - 1) - np.floor(np.log2(bmax + 1e-300))).astype(int)
    else:
        sftA = 6 - np.floor(np.log2(amax + 1e-300)).astype(int)
        sftB = 6 - np.floor(np.log2(bmax + 1e-300)).astype(int)
    
    Aprime = np.round(A * (2.0 ** sftA.reshape(-1, 1))).astype(np.int64)
    Bprime = np.round(B * (2.0 ** sftB.reshape(1, -1))).astype(np.int64)
    
    Cslices = []
    for pmod in moduli:
        Aslice = symmetric_mod(Aprime, pmod)
        Bslice = symmetric_mod(Bprime, pmod)
        Cprod = Aslice @ Bslice
        Cslices.append(symmetric_mod(Cprod, pmod))
    
    m, n = Cslices[0].shape
    Cscaled = np.zeros((m, n), dtype=np.float64)
    for row in range(m):
        for col in range(n):
            val = sum(crt_weights[i] * int(Cslices[i][row, col]) for i in range(num_mod))
            centered = val - round(val / M) * M
            Cscaled[row, col] = float(centered)
    
    C = np.diag(2.0 ** (-sftA)) @ Cscaled @ np.diag(2.0 ** (-sftB))
    return C

# Test matrices
A = np.array([[  1.111111111111111,   2.999999900000000],
              [ -5.123456767880000, -19.000011109999999]])
B = np.array([[ -3.500000000000000,   0.187500000000000],
              [  1.750000000000000,  -6.250000000000000]])
Ctrue = A @ B

print("="*70)
print("OZAKI II: Original (buggy) vs Fixed Implementation")
print("="*70)
print(f"\nTest matrices:")
print(f"A = {A.tolist()}")
print(f"B = {B.tolist()}")
print(f"True A*B =\n{Ctrue}\n")

print("-"*70)
print(f"{'num_mod':>8} | {'Original Error':>15} | {'Fixed Error':>15} | {'Improvement':>12}")
print("-"*70)

for num_mod in [2, 4, 6, 8, 10, 12]:
    C_orig = ozaki2(A, B, num_mod, fixed=False)
    C_fixed = ozaki2(A, B, num_mod, fixed=True)
    
    err_orig = np.linalg.norm(C_orig - Ctrue)
    err_fixed = np.linalg.norm(C_fixed - Ctrue)
    improvement = err_orig / (err_fixed + 1e-300)
    
    print(f"{num_mod:>8} | {err_orig:>15.6e} | {err_fixed:>15.6e} | {improvement:>10.0f}x")

print("-"*70)
print("\nBUGS FOUND IN emu_exp.m:")
print("  1. Moduli list missing 241 (goes 247->239 instead of 247->241->239)")
print("  2. Scaling uses hardcoded '6' instead of floor(log2(P-1)/2 - 1)")
print("\nFIXES APPLIED:")
print("  1. Added 241 to moduli_all")
print("  2. Changed sftA = 6 - floor(log2(amax))")
print("              to sftA = floor(log2(M-1)/2 - 1) - floor(log2(amax))")
print("="*70)
