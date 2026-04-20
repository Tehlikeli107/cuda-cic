/*
 * CIC Universe Level Evaluator: GPU-Native Universe Polymorphism
 * ===============================================================
 * Lean4 uses universe polymorphism: Sort u, where u can be:
 *   - zero
 *   - succ(u)
 *   - max(u, v)
 *   - imax(u, v)    // imax(u, 0) = 0, imax(u, succ(v)) = max(u, succ(v))
 *   - param(name)   // universe parameter variable
 *
 * Universe expressions are trees. We flatten them to GPU arrays
 * and evaluate bottom-up, just like expression type checking.
 *
 * For constants with universe parameters, we instantiate:
 *   @Eq.{u} → Eq where u is substituted with a concrete level
 *
 * Encoding:
 *   ulevel_kind[i]  = kind (ZERO/SUCC/MAX/IMAX/PARAM)
 *   ulevel_arg1[i]  = child1 index or param index
 *   ulevel_arg2[i]  = child2 index (for MAX/IMAX)
 *   ulevel_value[i] = evaluated concrete level (output)
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Universe level expression kinds
#define UL_ZERO    0
#define UL_SUCC    1
#define UL_MAX     2
#define UL_IMAX    3
#define UL_PARAM   4
#define UL_LIT     5  // pre-evaluated literal
#define UL_NONE   -1

// Limits
#define MAX_ULEVELS    256   // max universe level expressions per batch entry
#define MAX_UPARAMS     16   // max universe parameters per constant

// ============================================================
// DEVICE: Evaluate a universe level expression
// ============================================================

__device__ int64_t eval_ulevel_node(
    const int64_t* kinds,   // [MAX_ULEVELS]
    const int64_t* arg1s,   // [MAX_ULEVELS]
    const int64_t* arg2s,   // [MAX_ULEVELS]
    int64_t* values,        // [MAX_ULEVELS] — computed values
    const int64_t* param_subst,  // [MAX_UPARAMS] — param substitutions
    int idx,
    int n_levels
) {
    if (idx < 0 || idx >= n_levels) return 0;
    if (values[idx] >= 0) return values[idx];  // already computed

    int64_t kind = kinds[idx];
    int64_t result = 0;

    switch (kind) {
        case UL_ZERO:
            result = 0;
            break;

        case UL_LIT:
            result = arg1s[idx];
            break;

        case UL_SUCC: {
            int64_t child = arg1s[idx];
            int64_t child_val = (child >= 0 && child < n_levels) ?
                eval_ulevel_node(kinds, arg1s, arg2s, values, param_subst, (int)child, n_levels) : 0;
            result = child_val + 1;
            break;
        }

        case UL_MAX: {
            int64_t c1 = arg1s[idx], c2 = arg2s[idx];
            int64_t v1 = (c1 >= 0 && c1 < n_levels) ?
                eval_ulevel_node(kinds, arg1s, arg2s, values, param_subst, (int)c1, n_levels) : 0;
            int64_t v2 = (c2 >= 0 && c2 < n_levels) ?
                eval_ulevel_node(kinds, arg1s, arg2s, values, param_subst, (int)c2, n_levels) : 0;
            result = (v1 > v2) ? v1 : v2;
            break;
        }

        case UL_IMAX: {
            // imax(u, v):
            //   if v evaluates to 0 → result is 0
            //   otherwise → max(u, v)
            int64_t c1 = arg1s[idx], c2 = arg2s[idx];
            int64_t v1 = (c1 >= 0 && c1 < n_levels) ?
                eval_ulevel_node(kinds, arg1s, arg2s, values, param_subst, (int)c1, n_levels) : 0;
            int64_t v2 = (c2 >= 0 && c2 < n_levels) ?
                eval_ulevel_node(kinds, arg1s, arg2s, values, param_subst, (int)c2, n_levels) : 0;
            if (v2 == 0) {
                result = 0;
            } else {
                result = (v1 > v2) ? v1 : v2;
            }
            break;
        }

        case UL_PARAM: {
            int64_t param_idx = arg1s[idx];
            if (param_idx >= 0 && param_idx < MAX_UPARAMS) {
                result = param_subst[param_idx];
            }
            break;
        }

        default:
            result = 0;
    }

    values[idx] = result;
    return result;
}


// ============================================================
// KERNEL: Evaluate universe levels for a batch
// ============================================================

__global__ void eval_universe_levels_kernel(
    const int64_t* __restrict__ ulevel_kinds,   // [B, MAX_ULEVELS]
    const int64_t* __restrict__ ulevel_arg1,    // [B, MAX_ULEVELS]
    const int64_t* __restrict__ ulevel_arg2,    // [B, MAX_ULEVELS]
    int64_t* __restrict__ ulevel_values,        // [B, MAX_ULEVELS] — output
    const int64_t* __restrict__ param_subst,    // [B, MAX_UPARAMS]
    const int64_t* __restrict__ n_levels_per,   // [B]
    int B
) {
    int bi = blockIdx.x * blockDim.x + threadIdx.x;
    if (bi >= B) return;

    int base = bi * MAX_ULEVELS;
    int pbase = bi * MAX_UPARAMS;
    int nl = (int)n_levels_per[bi];
    if (nl <= 0 || nl > MAX_ULEVELS) return;

    // Initialize values to -1 (not computed)
    for (int i = 0; i < nl; i++) {
        ulevel_values[base + i] = -1;
    }

    // Evaluate all levels (recursive with memoization)
    for (int i = 0; i < nl; i++) {
        if (ulevel_kinds[base + i] != UL_NONE) {
            eval_ulevel_node(
                &ulevel_kinds[base],
                &ulevel_arg1[base],
                &ulevel_arg2[base],
                &ulevel_values[base],
                &param_subst[pbase],
                i, nl
            );
        }
    }
}


// ============================================================
// DEVICE: Sort type hash with proper universe level
// ============================================================
// Used by type checking kernel to compute Sort(u) : Sort(u+1)

#define T_PROP    1
#define T_TYPE    2
#define T_TYPE1   3
#define PRIME1    1000003LL
#define HASH_MOD  1048576LL

__device__ inline int64_t sort_hash_from_level(int64_t level) {
    if (level == 0) return T_PROP;    // Sort 0 = Prop
    if (level == 1) return T_TYPE;    // Sort 1 = Type
    if (level == 2) return T_TYPE1;   // Sort 2 = Type 1
    return ((level * PRIME1 + 0x60000000LL) % HASH_MOD) + HASH_MOD;
}

__device__ inline int64_t sort_type_of_level(int64_t level) {
    // Sort(u) : Sort(u+1)
    return sort_hash_from_level(level + 1);
}

__device__ inline int64_t pi_sort_level(int64_t dom_level, int64_t cod_level) {
    // Pi(A : Sort u, B : Sort v) : Sort(imax(u, v))
    // imax(u, 0) = 0, imax(u, v+1) = max(u, v+1)
    if (cod_level == 0) return 0;  // Prop-valued: result is Prop
    return (dom_level > cod_level) ? dom_level : cod_level;
}


// ============================================================
// HOST: Pipeline
// ============================================================

std::vector<torch::Tensor> eval_universe_levels(
    torch::Tensor ulevel_kinds,
    torch::Tensor ulevel_arg1,
    torch::Tensor ulevel_arg2,
    torch::Tensor param_subst,
    torch::Tensor n_levels_per
) {
    int B = ulevel_kinds.size(0);
    int threads = 256;
    int blocks = (B + threads - 1) / threads;

    auto values = torch::full({B, MAX_ULEVELS}, -1, ulevel_kinds.options());

    eval_universe_levels_kernel<<<blocks, threads>>>(
        ulevel_kinds.data_ptr<int64_t>(),
        ulevel_arg1.data_ptr<int64_t>(),
        ulevel_arg2.data_ptr<int64_t>(),
        values.data_ptr<int64_t>(),
        param_subst.data_ptr<int64_t>(),
        n_levels_per.data_ptr<int64_t>(),
        B
    );

    return {values};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("eval_universe_levels", &eval_universe_levels,
          "Evaluate universe level expressions on GPU");
}
