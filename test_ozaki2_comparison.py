#!/usr/bin/env python3
"""
Test Ozaki II implementation comparison between MATLAB-style (emu_exp) and GEMMul8
Uses simple 2x2 matrices to trace through all intermediate values

Compares three implementations:
1. ORIGINAL MATLAB emu_exp (buggy - hardcoded scaling, missing modulus 241)
2. FIXED MATLAB version (proper scaling, correct moduli)
3. GEMMul8-style (reference correct implementation)

Usage: python test_ozaki2_comparison.py [num_moduli]
"""

import numpy as np
from fractions import Fraction
import sys


def extended_gcd(a, b):
    """Extended Euclidean algorithm"""
    if a == 0:
        return b, 0, 1
    gcd, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return gcd, x, y


def modinv(a, m):
    """Modular inverse"""
    a = a % m
    if a == 0:
        return 0
    gcd, x, _ = extended_gcd(a, m)
    if gcd != 1:
        raise ValueError(f"Modular inverse doesn't exist for {a} mod {m}")
    return x % m


def symmetric_mod(x, p):
    """Symmetric modulo: maps to range [-(p-1)/2, p/2] for odd p, [-p/2, p/2-1] for even p"""
    r = x % p
    if isinstance(r, np.ndarray):
        r[r > p // 2] = r[r > p // 2] - p
    else:
        if r > p // 2:
            r = r - p
    return r


def emu_exp_debug(A, B, num_mod, fixed=False):
    """MATLAB emu_exp implementation with debug output
    
    Args:
        A, B: Input matrices
        num_mod: Number of moduli to use
        fixed: If True, use the fixed (correct) implementation
               If False, use the original buggy implementation
    """
    debug = {}
    
    if fixed:
        # FIXED moduli (includes 241!)
        moduli_all = [256, 255, 253, 251, 247, 241, 239, 233, 229, 227,
                      223, 217, 211, 199, 197, 193, 191, 181, 179, 173]
    else:
        # ORIGINAL BUGGY moduli (missing 241!)
        moduli_all = [256, 255, 253, 251, 247, 239, 233, 229, 
                      227, 223, 217, 211, 199, 197, 193, 191]
    moduli = moduli_all[:num_mod]
    debug['moduli'] = moduli
    
    p, nA = A.shape
    nB, r = B.shape
    assert nA == nB, 'Inner dimensions must match.'
    
    # Compute M (product of moduli)
    M = 1
    for m in moduli:
        M *= m
    debug['M'] = M
    
    # Compute CRT weights (symbolic precision using Python integers)
    crt_weights = []
    for i in range(num_mod):
        pi = moduli[i]
        Pi = M // pi
        Pi_mod = Pi % pi
        qi = modinv(Pi_mod, pi)
        crt_weights.append(qi * Pi)
    debug['crt_weights'] = crt_weights
    
    # Scaling
    amax = np.max(np.abs(A), axis=1)
    bmax = np.max(np.abs(B), axis=0)
    debug['amax'] = amax
    debug['bmax'] = bmax
    
    if fixed:
        # FIXED scaling formula (scales with P)
        log2P_half = np.log2(float(M - 1)) / 2.0
        sftA = (np.floor(log2P_half - 1) - np.floor(np.log2(amax + 1e-300))).astype(int)
        sftB = (np.floor(log2P_half - 1) - np.floor(np.log2(bmax + 1e-300))).astype(int)
    else:
        # ORIGINAL BUGGY scaling formula (hardcoded constant)
        sftA = 6 - np.floor(np.log2(amax + 1e-300)).astype(int)
        sftB = 6 - np.floor(np.log2(bmax + 1e-300)).astype(int)
    debug['sftA'] = sftA
    debug['sftB'] = sftB
    
    print(f"  sftA (row shifts) = {sftA}")
    print(f"  sftB (col shifts) = {sftB}")
    
    # Scale matrices
    Aprime = np.round(A * (2.0 ** sftA.reshape(-1, 1))).astype(np.int64)
    Bprime = np.round(B * (2.0 ** sftB.reshape(1, -1))).astype(np.int64)
    debug['Aprime'] = Aprime
    debug['Bprime'] = Bprime
    
    print(f"  Aprime (scaled A) =\n{Aprime}")
    print(f"  Bprime (scaled B) =\n{Bprime}")
    
    # Modular GEMMs
    Cslices = []
    Aslices = []
    Bslices = []
    Cprod_slices = []
    
    for i, pmod in enumerate(moduli):
        Aslice = symmetric_mod(Aprime, pmod)
        Bslice = symmetric_mod(Bprime, pmod)
        Cprod = Aslice @ Bslice
        Cslice = symmetric_mod(Cprod, pmod)
        
        Cslices.append(Cslice)
        Aslices.append(Aslice)
        Bslices.append(Bslice)
        Cprod_slices.append(Cprod)
        
        print(f"  Modulus {i+1} ({pmod}):")
        print(f"    Aslice = {Aslice}")
        print(f"    Bslice = {Bslice}")
        print(f"    Cprod (before mod) = {Cprod}")
        print(f"    Cslice (after mod) = {Cslice}")
    
    debug['Aslices'] = Aslices
    debug['Bslices'] = Bslices
    debug['Cprod_slices'] = Cprod_slices
    debug['Cslices'] = Cslices
    
    # CRT reconstruction (using Python integers for exact arithmetic)
    m, n = Cslices[0].shape
    Cscaled = np.zeros((m, n), dtype=np.float64)
    
    for row in range(m):
        for col in range(n):
            val = 0  # Python arbitrary precision integer
            for i in range(num_mod):
                val += crt_weights[i] * int(Cslices[i][row, col])
            # Center to [-M/2, M/2]
            centered = val - round(val / M) * M
            Cscaled[row, col] = float(centered)
    
    debug['Cscaled'] = Cscaled
    print(f"  Cscaled (after CRT) =\n{Cscaled}")
    
    # Inverse scaling
    C = np.diag(2.0 ** (-sftA)) @ Cscaled @ np.diag(2.0 ** (-sftB))
    debug['C'] = C
    
    print(f"  Final C (after inverse scaling) =\n{C}")
    
    return C, debug


def gemmul8_style(A, B, num_mod):
    """GEMMul8-style implementation with debug output"""
    debug = {}
    
    # GEMMul8 moduli (note: includes 241!)
    moduli_all = [256, 255, 253, 251, 247, 241, 239, 233, 229, 227,
                  223, 217, 211, 199, 197, 193, 191, 181, 179, 173]
    moduli = moduli_all[:num_mod]
    debug['moduli'] = moduli
    
    p, nA = A.shape
    nB, r = B.shape
    assert nA == nB, 'Inner dimensions must match.'
    
    # Compute P (product of moduli)
    P = 1
    for m in moduli:
        P *= m
    debug['P'] = P
    
    # Compute CRT weights
    qPi = []
    for i in range(num_mod):
        pi = moduli[i]
        Pi = P // pi
        Pi_mod = Pi % pi
        qi = modinv(Pi_mod, pi)
        qPi.append(float(qi * Pi))
    invP = 1.0 / float(P)
    debug['qPi'] = qPi
    debug['invP'] = invP
    
    # GEMMul8-style scaling (simplified)
    log2P_half = np.log2(float(P - 1)) / 2.0
    
    amax = np.max(np.abs(A), axis=1)
    bmax = np.max(np.abs(B), axis=0)
    debug['amax'] = amax
    debug['bmax'] = bmax
    
    sftA = (np.floor(log2P_half - 1) - np.floor(np.log2(amax + 1e-300))).astype(int)
    sftB = (np.floor(log2P_half - 1) - np.floor(np.log2(bmax + 1e-300))).astype(int)
    debug['sftA'] = sftA
    debug['sftB'] = sftB
    
    print(f"  sftA (row shifts) = {sftA}")
    print(f"  sftB (col shifts) = {sftB}")
    
    # Scale matrices
    Aprime = np.zeros_like(A, dtype=np.int64)
    for i in range(A.shape[0]):
        Aprime[i, :] = np.round(A[i, :] * (2.0 ** sftA[i]))
    
    Bprime = np.zeros_like(B, dtype=np.int64)
    for j in range(B.shape[1]):
        Bprime[:, j] = np.round(B[:, j] * (2.0 ** sftB[j]))
    
    debug['Aprime'] = Aprime
    debug['Bprime'] = Bprime
    
    print(f"  Aprime (scaled A) =\n{Aprime}")
    print(f"  Bprime (scaled B) =\n{Bprime}")
    
    # Modular GEMMs
    Cslices = []
    Aslices = []
    Bslices = []
    Cprod_slices = []
    
    for i, pmod in enumerate(moduli):
        Aslice = symmetric_mod(Aprime, pmod)
        Bslice = symmetric_mod(Bprime, pmod)
        Cprod = Aslice @ Bslice
        Cslice = symmetric_mod(Cprod, pmod)
        
        Cslices.append(Cslice)
        Aslices.append(Aslice)
        Bslices.append(Bslice)
        Cprod_slices.append(Cprod)
        
        print(f"  Modulus {i+1} ({pmod}):")
        print(f"    Aslice = {Aslice}")
        print(f"    Bslice = {Bslice}")
        print(f"    Cprod (before mod) = {Cprod}")
        print(f"    Cslice (after mod) = {Cslice}")
    
    debug['Aslices'] = Aslices
    debug['Bslices'] = Bslices
    debug['Cprod_slices'] = Cprod_slices
    debug['Cslices'] = Cslices
    
    # CRT reconstruction (GEMMul8 style - using floating point)
    m, n = Cslices[0].shape
    Cscaled = np.zeros((m, n), dtype=np.float64)
    P_double = float(P)
    
    for row in range(m):
        for col in range(n):
            val = 0.0
            for i in range(num_mod):
                val += qPi[i] * float(Cslices[i][row, col])
            # Center: val - round(val/P)*P
            quot = round(invP * val)
            centered = val - P_double * quot
            Cscaled[row, col] = centered
    
    debug['Cscaled'] = Cscaled
    print(f"  Cscaled (after CRT) =\n{Cscaled}")
    
    # Inverse scaling
    C = np.zeros_like(Cscaled)
    for i in range(m):
        for j in range(n):
            C[i, j] = Cscaled[i, j] * (2.0 ** (-sftA[i])) * (2.0 ** (-sftB[j]))
    
    debug['C'] = C
    print(f"  Final C (after inverse scaling) =\n{C}")
    
    return C, debug


def compare_intermediates(matlab_debug, gemmul8_debug):
    """Compare intermediate values between implementations"""
    
    print("\n1. MODULI COMPARISON:")
    print(f"   MATLAB:  {matlab_debug['moduli']}")
    print(f"   GEMMul8: {gemmul8_debug['moduli']}")
    if matlab_debug['moduli'] != gemmul8_debug['moduli']:
        print("   *** DIFFERENCE: Moduli lists differ! ***")
        print("   This is a KEY difference.")
        print("   MATLAB uses: 256, 255, 253, 251, 247, 239, ...")
        print("   GEMMul8 uses: 256, 255, 253, 251, 247, 241, 239, ...")
        print("   MATLAB is MISSING 241!")
    
    print("\n2. SCALING SHIFTS COMPARISON:")
    print(f"   MATLAB sftA:  {matlab_debug['sftA']}")
    print(f"   GEMMul8 sftA: {gemmul8_debug['sftA']}")
    if not np.array_equal(matlab_debug['sftA'], gemmul8_debug['sftA']):
        print("   *** DIFFERENCE in sftA ***")
    
    print(f"   MATLAB sftB:  {matlab_debug['sftB']}")
    print(f"   GEMMul8 sftB: {gemmul8_debug['sftB']}")
    if not np.array_equal(matlab_debug['sftB'], gemmul8_debug['sftB']):
        print("   *** DIFFERENCE in sftB ***")
    
    print("\n3. SCALED MATRICES COMPARISON:")
    print(f"   MATLAB Aprime:\n{matlab_debug['Aprime']}")
    print(f"   GEMMul8 Aprime:\n{gemmul8_debug['Aprime']}")
    if not np.array_equal(matlab_debug['Aprime'], gemmul8_debug['Aprime']):
        print("   *** DIFFERENCE in Aprime ***")
        print(f"   Difference:\n{matlab_debug['Aprime'] - gemmul8_debug['Aprime']}")
    
    print(f"\n   MATLAB Bprime:\n{matlab_debug['Bprime']}")
    print(f"   GEMMul8 Bprime:\n{gemmul8_debug['Bprime']}")
    if not np.array_equal(matlab_debug['Bprime'], gemmul8_debug['Bprime']):
        print("   *** DIFFERENCE in Bprime ***")
        print(f"   Difference:\n{matlab_debug['Bprime'] - gemmul8_debug['Bprime']}")
    
    print("\n4. MODULAR SLICES COMPARISON:")
    num_mod = len(matlab_debug['moduli'])
    for i in range(num_mod):
        print(f"   Modulus {i+1}:")
        print(f"     MATLAB Aslice:  {matlab_debug['Aslices'][i]}")
        print(f"     GEMMul8 Aslice: {gemmul8_debug['Aslices'][i]}")
        if not np.array_equal(matlab_debug['Aslices'][i], gemmul8_debug['Aslices'][i]):
            print("     *** DIFFERENCE in Aslice ***")
        print(f"     MATLAB Cslice:  {matlab_debug['Cslices'][i]}")
        print(f"     GEMMul8 Cslice: {gemmul8_debug['Cslices'][i]}")
        if not np.array_equal(matlab_debug['Cslices'][i], gemmul8_debug['Cslices'][i]):
            print("     *** DIFFERENCE in Cslice ***")
    
    print("\n5. CRT RECONSTRUCTION COMPARISON:")
    print(f"   MATLAB Cscaled:\n{matlab_debug['Cscaled']}")
    print(f"   GEMMul8 Cscaled:\n{gemmul8_debug['Cscaled']}")
    if np.linalg.norm(matlab_debug['Cscaled'] - gemmul8_debug['Cscaled']) > 1e-10:
        print("   *** DIFFERENCE in Cscaled ***")
        print(f"   Difference:\n{matlab_debug['Cscaled'] - gemmul8_debug['Cscaled']}")
    
    print("\n6. FINAL RESULT COMPARISON:")
    print(f"   MATLAB C:\n{matlab_debug['C']}")
    print(f"   GEMMul8 C:\n{gemmul8_debug['C']}")
    print(f"   Difference (C_matlab - C_gemmul8):\n{matlab_debug['C'] - gemmul8_debug['C']}")


def main():
    num_mod = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    
    print("\n" + "="*60)
    print(f"Ozaki II Comparison Test (num_mod = {num_mod})")
    print("="*60)
    
    # Test matrices (user provided)
    A = np.array([[  1.111111111111111,   2.999999900000000],
                  [ -5.123456767880000, -19.000011109999999]])
    B = np.array([[ -3.500000000000000,   0.187500000000000],
                  [  1.750000000000000,  -6.250000000000000]])
    
    print("\nInput matrices:")
    print(f"A =\n{A}")
    print(f"B =\n{B}")
    
    # True result
    Ctrue = A @ B
    print(f"\nTrue A*B =\n{Ctrue}")
    
    print("\n" + "-"*60)
    print("Running ORIGINAL MATLAB emu_exp (BUGGY)")
    print("-"*60)
    C_matlab, matlab_debug = emu_exp_debug(A, B, num_mod, fixed=False)
    
    print("\n" + "-"*60)
    print("Running FIXED MATLAB emu_exp")
    print("-"*60)
    C_fixed, fixed_debug = emu_exp_debug(A, B, num_mod, fixed=True)
    
    print("\n" + "-"*60)
    print("Running GEMMul8-style (CORRECT reference)")
    print("-"*60)
    C_gemmul8, gemmul8_debug = gemmul8_style(A, B, num_mod)
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    print(f"\nTrue C (A*B):\n{Ctrue}")
    print(f"\nOriginal MATLAB emu_exp C (BUGGY):\n{C_matlab}")
    print(f"\nFIXED MATLAB emu_exp C:\n{C_fixed}")
    print(f"\nGEMMul8-style C (CORRECT):\n{C_gemmul8}")
    
    err_matlab = np.linalg.norm(C_matlab - Ctrue)
    err_fixed = np.linalg.norm(C_fixed - Ctrue)
    err_gemmul8 = np.linalg.norm(C_gemmul8 - Ctrue)
    rel_err_matlab = err_matlab / (np.linalg.norm(Ctrue) + np.finfo(float).eps)
    rel_err_fixed = err_fixed / (np.linalg.norm(Ctrue) + np.finfo(float).eps)
    rel_err_gemmul8 = err_gemmul8 / (np.linalg.norm(Ctrue) + np.finfo(float).eps)
    
    print("\nError norms:")
    print(f"  Original MATLAB:  {err_matlab:.16e}")
    print(f"  FIXED MATLAB:     {err_fixed:.16e}")
    print(f"  GEMMul8-style:    {err_gemmul8:.16e}")
    print(f"  Improvement:      {err_matlab / (err_fixed + 1e-300):.1f}x better with fix")
    
    print("\nRelative errors:")
    print(f"  Original MATLAB:  {rel_err_matlab:.16e}")
    print(f"  FIXED MATLAB:     {rel_err_fixed:.16e}")
    print(f"  GEMMul8-style:    {rel_err_gemmul8:.16e}")
    
    # Verify fixed matches GEMMul8
    if np.linalg.norm(C_fixed - C_gemmul8) < 1e-10:
        print("\n*** FIXED MATLAB matches GEMMul8 (SUCCESS!) ***")
    
    print("\n" + "="*60)
    print("BUGS FOUND IN emu_exp.m")
    print("="*60)
    
    print("\n*** BUG 1: WRONG MODULI LIST ***")
    print(f"   MATLAB moduli:   {matlab_debug['moduli']}")
    print(f"   GEMMul8 moduli:  {gemmul8_debug['moduli']}")
    if matlab_debug['moduli'] != gemmul8_debug['moduli']:
        print("   PROBLEM: MATLAB is MISSING modulus 241!")
        print("   MATLAB goes: 247 -> 239")
        print("   Should be:   247 -> 241 -> 239")
    
    print("\n*** BUG 2: HARDCODED SCALING FORMULA ***")
    print(f"   MATLAB uses:   sftA = 6 - floor(log2(amax))")
    print(f"   GEMMul8 uses:  sftA = floor(log2(P-1)/2 - 1) - floor(log2(amax))")
    print(f"   ")
    print(f"   For num_mod={num_mod}:")
    matlab_P = 1
    for m in matlab_debug['moduli']:
        matlab_P *= m
    gemmul8_P = 1
    for m in gemmul8_debug['moduli']:
        gemmul8_P *= m
    print(f"     MATLAB P = {matlab_P:,}")
    print(f"     GEMMul8 P = {gemmul8_P:,}")
    print(f"     MATLAB log2(P)/2 = {np.log2(float(matlab_P))/2:.2f}")
    print(f"     GEMMul8 log2(P)/2 = {np.log2(float(gemmul8_P))/2:.2f}")
    print(f"     MATLAB sftA:  {matlab_debug['sftA']}")
    print(f"     GEMMul8 sftA: {gemmul8_debug['sftA']}")
    
    print("\n*** FIX REQUIRED IN emu_exp.m ***")
    print("   Line 3: Change moduli_all to include 241:")
    print("     moduli_all = [256, 255, 253, 251, 247, 241, 239, ...];")
    print("")
    print("   Lines 16-17: Change scaling formula:")
    print("     % WRONG:")
    print("     %   sftA = 6 - floor(log2(amax));")
    print("     %   sftB = 6 - floor(log2(bmax));")
    print("     % CORRECT:")
    print("     log2P_half = log2(double(M - 1)) / 2;")
    print("     sftA = floor(log2P_half - 1) - floor(log2(amax));")
    print("     sftB = floor(log2P_half - 1) - floor(log2(bmax));")
    
    print("\n" + "="*60)
    print("STEP-BY-STEP COMPARISON")
    print("="*60)
    
    compare_intermediates(matlab_debug, gemmul8_debug)


if __name__ == "__main__":
    main()
