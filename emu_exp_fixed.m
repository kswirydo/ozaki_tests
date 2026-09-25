function C = emu_exp_fixed(A, B, num_mod)
% Ozaki Scheme II - FIXED version matching GEMMul8
% 
% Key fixes compared to original emu_exp.m:
%   1. Moduli list now includes 241 (was missing in original)
%   2. Scaling formula now uses log2(P) instead of hardcoded constant
%
% ORIGINAL BUG: The scaling formula was hardcoded as:
%   sftA = 6 - floor(log2(amax))
% This assumes only 2 moduli (P = 256*255 = 65280, log2(P)/2 ≈ 8)
%
% FIXED: Now uses:
%   sftA = floor(log2(P-1)/2 - 1) - floor(log2(amax))
% This scales properly with the number of moduli

% GEMMul8 moduli (includes 241!)
moduli_all = [256, 255, 253, 251, 247, 241, 239, 233, 229, 227, ...
              223, 217, 211, 199, 197, 193, 191, 181, 179, 173];
moduli = moduli_all(1:num_mod);
[p, nA] = size(A);
[nB, r] = size(B);
assert(nA == nB, 'Inner dimensions must match.');

% Precompute CRT weights ONCE (symbolic)
[crt_weights, M] = precompute_crt_weights_sym(moduli);

% ----------------------------
% Scaling (FIXED - scales with P)
% ----------------------------
log2P_half = log2(double(M - 1)) / 2;

amax = max(abs(A), [], 2);
bmax = max(abs(B), [], 1);

% GEMMul8 formula: floor(log2(P-1)/2 - 1) - floor(log2(amax))
sftA = floor(log2P_half - 1) - floor(log2(amax));
sftB = floor(log2P_half - 1) - floor(log2(bmax));
sftA(~isfinite(sftA)) = 0;
sftB(~isfinite(sftB)) = 0;

Aprime = round(A .* (2 .^ sftA));
Bprime = round(B .* (2 .^ sftB));

% ----------------------------
% Modular GEMMs
% ----------------------------
Cslices = cell(num_mod, 1);
for i = 1:num_mod
    pmod = moduli(i);
    Aslice = symmetric_mod(Aprime, pmod);
    Bslice = symmetric_mod(Bprime, pmod);
    Cprod = Aslice * Bslice;
    Cslices{i} = symmetric_mod(Cprod, pmod);
end

% ----------------------------
% CRT reconstruction (symbolic, but optimized)
% ----------------------------
Cscaled = crt_reconstruct_sym(Cslices, crt_weights, M);

% ----------------------------
% Inverse scaling
% ----------------------------
C = diag(2 .^ (-sftA)) * Cscaled * diag(2 .^ (-sftB));

Ctrue = A * B;
fprintf("Error norm: %16.16e\n", norm(C - Ctrue));
fprintf("Relative error: %16.16e\n", norm(C - Ctrue) / (norm(Ctrue) + eps));
end

function r = symmetric_mod(x, p)
r = mod(x, p);
r(r > p/2) = r(r > p/2) - p;
end

function [weights, M] = precompute_crt_weights_sym(moduli)
% Precompute symbolic CRT weights - call once, reuse many times
num_mod = numel(moduli);
% Use sym for exact integer arithmetic
M = prod(sym(moduli));
weights = sym(zeros(num_mod, 1));
for i = 1:num_mod
    pi = sym(moduli(i));
    Pi = M / pi;                              % Product of other moduli
    Pi_mod = double(mod(Pi, pi));             % Small, fits in double
    qi = modinv(Pi_mod, double(pi));          % Modular inverse
    weights(i) = sym(qi) * Pi;
end
end

function X = crt_reconstruct_sym(C, weights, M)
% Optimized symbolic CRT using precomputed weights
num_mod = numel(C);
[m, n] = size(C{1});
X = zeros(m, n);
for row = 1:m
    for col = 1:n
        val = sym(0);
        for i = 1:num_mod
            val = val + weights(i) * sym(C{i}(row, col));
        end
        % Center to [-M/2, M/2]
        val = val - round(val / M) * M;
        X(row, col) = double(val);
    end
end
end

function inv = modinv(a, m)
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
