function test_ozaki2_comparison(num_mod)
% Test Ozaki II implementation comparison between MATLAB (emu_exp) and GEMMul8
% Uses simple 2x2 matrices to trace through all intermediate values
%
% Usage: test_ozaki2_comparison(num_mod)
%   num_mod: number of moduli to use (2-16)

if nargin < 1
    num_mod = 2;
end

fprintf('\n========================================\n');
fprintf('Ozaki II Comparison Test (num_mod = %d)\n', num_mod);
fprintf('========================================\n\n');

% Test matrices (user provided)
A = [   1.111111111111111   2.999999900000000
      -5.123456767880000 -19.000011109999999];
B = [  -3.500000000000000   0.187500000000000
        1.750000000000000  -6.250000000000000];

fprintf('Input matrices:\n');
fprintf('A =\n');
disp(A);
fprintf('B =\n');
disp(B);

% True result (high precision)
Ctrue = A * B;
fprintf('True A*B =\n');
disp(Ctrue);

% Run both implementations and compare
fprintf('\n--- Running MATLAB emu_exp implementation ---\n');
[C_matlab, matlab_debug] = emu_exp_debug(A, B, num_mod);

fprintf('\n--- Running GEMMul8-style implementation ---\n');
[C_gemmul8, gemmul8_debug] = gemmul8_style(A, B, num_mod);

% Compare results
fprintf('\n========================================\n');
fprintf('RESULTS COMPARISON\n');
fprintf('========================================\n\n');

fprintf('True C (A*B):\n');
disp(Ctrue);

fprintf('MATLAB emu_exp C:\n');
disp(C_matlab);

fprintf('GEMMul8-style C:\n');
disp(C_gemmul8);

fprintf('Error norms:\n');
fprintf('  MATLAB emu_exp:   %16.16e\n', norm(C_matlab - Ctrue));
fprintf('  GEMMul8-style:    %16.16e\n', norm(C_gemmul8 - Ctrue));

fprintf('\nRelative errors:\n');
fprintf('  MATLAB emu_exp:   %16.16e\n', norm(C_matlab - Ctrue) / (norm(Ctrue) + eps));
fprintf('  GEMMul8-style:    %16.16e\n', norm(C_gemmul8 - Ctrue) / (norm(Ctrue) + eps));

% Compare intermediate values
fprintf('\n========================================\n');
fprintf('STEP-BY-STEP COMPARISON\n');
fprintf('========================================\n\n');

compare_intermediates(matlab_debug, gemmul8_debug);

end

function [C, debug] = emu_exp_debug(A, B, num_mod)
% MATLAB emu_exp implementation with debug output
% This is essentially the same as emu_exp.m but returns intermediate values

moduli_all = [256, 255, 253, 251, 247, 239, 233, 229, ...
              227, 223, 217, 211, 199, 197, 193, 191];
moduli = moduli_all(1:num_mod);
debug.moduli = moduli;

[p, nA] = size(A);
[nB, r] = size(B);
assert(nA == nB, 'Inner dimensions must match.');

% Precompute CRT weights
[crt_weights, M] = precompute_crt_weights_sym(moduli);
debug.crt_weights = crt_weights;
debug.M = M;

% Scaling
amax = max(abs(A), [], 2);
bmax = max(abs(B), [], 1);
debug.amax = amax;
debug.bmax = bmax;

sftA = 6 - floor(log2(amax));
sftB = 6 - floor(log2(bmax));
sftA(~isfinite(sftA)) = 0;
sftB(~isfinite(sftB)) = 0;
debug.sftA = sftA;
debug.sftB = sftB;

fprintf('  sftA (row shifts) = %s\n', mat2str(sftA'));
fprintf('  sftB (col shifts) = %s\n', mat2str(sftB));

Aprime = round(A .* (2 .^ sftA));
Bprime = round(B .* (2 .^ sftB));
debug.Aprime = Aprime;
debug.Bprime = Bprime;

fprintf('  Aprime (scaled A) =\n');
disp(Aprime);
fprintf('  Bprime (scaled B) =\n');
disp(Bprime);

% Modular GEMMs
Cslices = cell(num_mod, 1);
Aslices = cell(num_mod, 1);
Bslices = cell(num_mod, 1);
Cprod_slices = cell(num_mod, 1);

for i = 1:num_mod
    pmod = moduli(i);
    Aslice = symmetric_mod(Aprime, pmod);
    Bslice = symmetric_mod(Bprime, pmod);
    Cprod = Aslice * Bslice;
    Cslices{i} = symmetric_mod(Cprod, pmod);
    
    Aslices{i} = Aslice;
    Bslices{i} = Bslice;
    Cprod_slices{i} = Cprod;
    
    fprintf('  Modulus %d (%d):\n', i, pmod);
    fprintf('    Aslice = %s\n', mat2str(Aslice));
    fprintf('    Bslice = %s\n', mat2str(Bslice));
    fprintf('    Cprod (before mod) = %s\n', mat2str(Cprod));
    fprintf('    Cslice (after mod) = %s\n', mat2str(Cslices{i}));
end

debug.Aslices = Aslices;
debug.Bslices = Bslices;
debug.Cprod_slices = Cprod_slices;
debug.Cslices = Cslices;

% CRT reconstruction
Cscaled = crt_reconstruct_sym_debug(Cslices, crt_weights, M, debug);
debug.Cscaled = Cscaled;

fprintf('  Cscaled (after CRT) =\n');
disp(Cscaled);

% Inverse scaling
C = diag(2 .^ (-sftA)) * Cscaled * diag(2 .^ (-sftB));
debug.C = C;

fprintf('  Final C (after inverse scaling) =\n');
disp(C);

end

function [C, debug] = gemmul8_style(A, B, num_mod)
% GEMMul8-style implementation with debug output
% Mimics the GEMMul8 algorithm more closely

% GEMMul8 moduli (note: starts with 256, then 255, 253, etc.)
moduli_all = [256, 255, 253, 251, 247, 241, 239, 233, 229, 227, ...
              223, 217, 211, 199, 197, 193, 191, 181, 179, 173];
moduli = moduli_all(1:num_mod);
debug.moduli = moduli;

[p, nA] = size(A);
[nB, r] = size(B);
assert(nA == nB, 'Inner dimensions must match.');

% Compute P (product of moduli)
P = prod(sym(moduli));
debug.P = P;

% Compute CRT weights (qPi values)
[qPi, invP] = compute_gemmul8_crt_weights(moduli);
debug.qPi = qPi;
debug.invP = invP;

% GEMMul8-style scaling
% Uses more sophisticated formula based on max and vector norm
[sftA, sftB] = gemmul8_scaling(A, B, P);
debug.sftA = sftA;
debug.sftB = sftB;

fprintf('  sftA (row shifts) = %s\n', mat2str(sftA'));
fprintf('  sftB (col shifts) = %s\n', mat2str(sftB));

% Scale and round to integers
Aprime = zeros(size(A));
for i = 1:size(A, 1)
    Aprime(i, :) = round(A(i, :) * (2^sftA(i)));
end
Bprime = zeros(size(B));
for j = 1:size(B, 2)
    Bprime(:, j) = round(B(:, j) * (2^sftB(j)));
end
debug.Aprime = Aprime;
debug.Bprime = Bprime;

fprintf('  Aprime (scaled A) =\n');
disp(Aprime);
fprintf('  Bprime (scaled B) =\n');
disp(Bprime);

% Modular GEMMs
Cslices = cell(num_mod, 1);
Aslices = cell(num_mod, 1);
Bslices = cell(num_mod, 1);
Cprod_slices = cell(num_mod, 1);

for i = 1:num_mod
    pmod = moduli(i);
    Aslice = gemmul8_symmetric_mod(Aprime, pmod);
    Bslice = gemmul8_symmetric_mod(Bprime, pmod);
    Cprod = Aslice * Bslice;
    Cslices{i} = gemmul8_symmetric_mod(Cprod, pmod);
    
    Aslices{i} = Aslice;
    Bslices{i} = Bslice;
    Cprod_slices{i} = Cprod;
    
    fprintf('  Modulus %d (%d):\n', i, pmod);
    fprintf('    Aslice = %s\n', mat2str(Aslice));
    fprintf('    Bslice = %s\n', mat2str(Bslice));
    fprintf('    Cprod (before mod) = %s\n', mat2str(Cprod));
    fprintf('    Cslice (after mod) = %s\n', mat2str(Cslices{i}));
end

debug.Aslices = Aslices;
debug.Bslices = Bslices;
debug.Cprod_slices = Cprod_slices;
debug.Cslices = Cslices;

% CRT reconstruction (GEMMul8 style)
Cscaled = gemmul8_crt_reconstruct(Cslices, qPi, P, invP);
debug.Cscaled = Cscaled;

fprintf('  Cscaled (after CRT) =\n');
disp(Cscaled);

% Inverse scaling
C = zeros(size(Cscaled));
for i = 1:size(C, 1)
    for j = 1:size(C, 2)
        C(i, j) = Cscaled(i, j) * 2^(-sftA(i)) * 2^(-sftB(j));
    end
end
debug.C = C;

fprintf('  Final C (after inverse scaling) =\n');
disp(C);

end

function [qPi, invP] = compute_gemmul8_crt_weights(moduli)
% Compute CRT weights like GEMMul8 does
num_mod = numel(moduli);
P = prod(sym(moduli));
invP = double(1/P);

qPi = zeros(num_mod, 1);
for i = 1:num_mod
    pi = sym(moduli(i));
    Pi = P / pi;
    Pi_mod = double(mod(Pi, pi));
    qi = modinv(Pi_mod, double(pi));
    qPi(i) = double(sym(qi) * Pi);
end
end

function [sftA, sftB] = gemmul8_scaling(A, B, P)
% GEMMul8-style scaling (simplified version)
% Uses: sft = floor(log2(P-1)/2) - 1 - floor(log2(amax))
% This is approximately: 6 - floor(log2(amax)) for small num_mod

log2P_half = log2(double(P - 1)) / 2;

% Row-wise for A
amax = max(abs(A), [], 2);
sftA = floor(log2P_half - 1) - floor(log2(amax));
sftA(~isfinite(sftA)) = 0;

% Column-wise for B
bmax = max(abs(B), [], 1)';
sftB = floor(log2P_half - 1) - floor(log2(bmax));
sftB(~isfinite(sftB)) = 0;
end

function r = symmetric_mod(x, p)
% Symmetric mod as in emu_exp.m
r = mod(x, p);
r(r > p/2) = r(r > p/2) - p;
end

function r = gemmul8_symmetric_mod(x, p)
% GEMMul8-style symmetric mod
% For mod 256, result is in [-128, 127]
% For other mods, result is in [-(p-1)/2, (p-1)/2]
r = mod(x, p);
r(r > p/2) = r(r > p/2) - p;
end

function [weights, M] = precompute_crt_weights_sym(moduli)
% Precompute symbolic CRT weights
num_mod = numel(moduli);
M = prod(sym(moduli));
weights = sym(zeros(num_mod, 1));
for i = 1:num_mod
    pi = sym(moduli(i));
    Pi = M / pi;
    Pi_mod = double(mod(Pi, pi));
    qi = modinv(Pi_mod, double(pi));
    weights(i) = sym(qi) * Pi;
end
end

function X = crt_reconstruct_sym_debug(C, weights, M, debug)
% CRT reconstruction with debug output
num_mod = numel(C);
[m, n] = size(C{1});
X = zeros(m, n);
for row = 1:m
    for col = 1:n
        val = sym(0);
        for i = 1:num_mod
            contribution = weights(i) * sym(C{i}(row, col));
            val = val + contribution;
        end
        % Center to [-M/2, M/2]
        centered = val - round(val / M) * M;
        X(row, col) = double(centered);
    end
end
end

function X = gemmul8_crt_reconstruct(C, qPi, P, invP)
% GEMMul8-style CRT reconstruction
num_mod = numel(C);
[m, n] = size(C{1});
X = zeros(m, n);

P_double = double(P);

for row = 1:m
    for col = 1:n
        % Sum: sum(qPi(i) * C{i}(row,col))
        val = 0;
        for i = 1:num_mod
            val = val + qPi(i) * C{i}(row, col);
        end
        % Center: val - round(val/P)*P
        quot = round(invP * val);
        centered = val - P_double * quot;
        X(row, col) = centered;
    end
end
end

function inv = modinv(a, m)
% Modular inverse using extended Euclidean algorithm
a = mod(a, m);
if a == 0
    inv = 0;
    return;
end
r0 = m; r1 = a;
s0 = 0; s1 = 1;
while r1 ~= 0
    q = floor(r0 / r1);
    [r0, r1] = deal(r1, r0 - q * r1);
    [s0, s1] = deal(s1, s0 - q * s1);
end
inv = mod(s0, m);
end

function compare_intermediates(matlab_debug, gemmul8_debug)
% Compare intermediate values between the two implementations

fprintf('1. MODULI COMPARISON:\n');
fprintf('   MATLAB:  %s\n', mat2str(matlab_debug.moduli));
fprintf('   GEMMul8: %s\n', mat2str(gemmul8_debug.moduli));
if ~isequal(matlab_debug.moduli, gemmul8_debug.moduli)
    fprintf('   *** DIFFERENCE: Moduli lists differ! ***\n');
    fprintf('   This is the FIRST key difference.\n');
    fprintf('   MATLAB uses: 256, 255, 253, 251, 247, 239, ...\n');
    fprintf('   GEMMul8 uses: 256, 255, 253, 251, 247, 241, 239, ...\n');
end
fprintf('\n');

fprintf('2. SCALING SHIFTS COMPARISON:\n');
fprintf('   MATLAB sftA:  %s\n', mat2str(matlab_debug.sftA'));
fprintf('   GEMMul8 sftA: %s\n', mat2str(gemmul8_debug.sftA'));
if ~isequal(matlab_debug.sftA, gemmul8_debug.sftA)
    fprintf('   *** DIFFERENCE in sftA ***\n');
end
fprintf('   MATLAB sftB:  %s\n', mat2str(matlab_debug.sftB));
fprintf('   GEMMul8 sftB: %s\n', mat2str(gemmul8_debug.sftB));
if ~isequal(matlab_debug.sftB, gemmul8_debug.sftB)
    fprintf('   *** DIFFERENCE in sftB ***\n');
end
fprintf('\n');

fprintf('3. SCALED MATRICES COMPARISON:\n');
fprintf('   MATLAB Aprime:\n');
disp(matlab_debug.Aprime);
fprintf('   GEMMul8 Aprime:\n');
disp(gemmul8_debug.Aprime);
if ~isequal(matlab_debug.Aprime, gemmul8_debug.Aprime)
    fprintf('   *** DIFFERENCE in Aprime ***\n');
    fprintf('   Difference: \n');
    disp(matlab_debug.Aprime - gemmul8_debug.Aprime);
end

fprintf('   MATLAB Bprime:\n');
disp(matlab_debug.Bprime);
fprintf('   GEMMul8 Bprime:\n');
disp(gemmul8_debug.Bprime);
if ~isequal(matlab_debug.Bprime, gemmul8_debug.Bprime)
    fprintf('   *** DIFFERENCE in Bprime ***\n');
    fprintf('   Difference: \n');
    disp(matlab_debug.Bprime - gemmul8_debug.Bprime);
end
fprintf('\n');

fprintf('4. MODULAR SLICES COMPARISON:\n');
num_mod = numel(matlab_debug.moduli);
for i = 1:num_mod
    fprintf('   Modulus %d:\n', i);
    fprintf('     MATLAB Aslice:  %s\n', mat2str(matlab_debug.Aslices{i}));
    fprintf('     GEMMul8 Aslice: %s\n', mat2str(gemmul8_debug.Aslices{i}));
    if ~isequal(matlab_debug.Aslices{i}, gemmul8_debug.Aslices{i})
        fprintf('     *** DIFFERENCE in Aslice ***\n');
    end
    fprintf('     MATLAB Cslice:  %s\n', mat2str(matlab_debug.Cslices{i}));
    fprintf('     GEMMul8 Cslice: %s\n', mat2str(gemmul8_debug.Cslices{i}));
    if ~isequal(matlab_debug.Cslices{i}, gemmul8_debug.Cslices{i})
        fprintf('     *** DIFFERENCE in Cslice ***\n');
    end
end
fprintf('\n');

fprintf('5. CRT RECONSTRUCTION COMPARISON:\n');
fprintf('   MATLAB Cscaled:\n');
disp(matlab_debug.Cscaled);
fprintf('   GEMMul8 Cscaled:\n');
disp(gemmul8_debug.Cscaled);
if norm(matlab_debug.Cscaled - gemmul8_debug.Cscaled) > 1e-10
    fprintf('   *** DIFFERENCE in Cscaled ***\n');
    fprintf('   Difference: \n');
    disp(matlab_debug.Cscaled - gemmul8_debug.Cscaled);
end
fprintf('\n');

fprintf('6. FINAL RESULT COMPARISON:\n');
fprintf('   MATLAB C:\n');
disp(matlab_debug.C);
fprintf('   GEMMul8 C:\n');
disp(gemmul8_debug.C);
fprintf('   Difference (C_matlab - C_gemmul8):\n');
disp(matlab_debug.C - gemmul8_debug.C);

end
