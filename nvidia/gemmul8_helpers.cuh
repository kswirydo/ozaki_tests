#pragma once

#include "worksize.hpp"

/** Disambiguate gemmul8::workSize for INT8 GEMM (two template overloads match a 4-arg call). */
inline size_t gemmul8_gemm_worksize(size_t m, size_t n, size_t k, int num_moduli) {
    return gemmul8::workSize<false, gemmul8::Backend::INT8, gemmul8::Func::gemm>(
        m, n, k, num_moduli);
}
