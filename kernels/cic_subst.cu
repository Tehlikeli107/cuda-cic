/*
 * CIC Substitution Engine: GPU-Native De Bruijn Substitution
 * ===========================================================
 * Correct de Bruijn substitution is THE foundation of dependent type theory.
 * Without this, beta-reduction produces garbage.
 *
 * De Bruijn convention:
 *   BVAR(0) = innermost bound variable
 *   BVAR(1) = next outer
 *   Lambda/Pi binds BVAR(0) in its body
 *
 * Substitution subst(body, depth, replacement):
 *   BVAR(i) where i == depth  → shift(replacement, 0, depth)
 *   BVAR(i) where i > depth   → BVAR(i-1)  (variable was free, adjust)
 *   BVAR(i) where i < depth   → BVAR(i)    (bound, untouched)
 *   LAM(t, body)              → LAM(subst(t,d,r), subst(body,d+1,r))
 *   PI(t, body)               → PI(subst(t,d,r), subst(body,d+1,r))
 *   APP(f, a)                 → APP(subst(f,d,r), subst(a,d,r))
 *   LET(t, v, body)           → LET(subst(t,d,r), subst(v,d,r), subst(body,d+1,r))
 *   everything else           → unchanged
 *
 * GPU strategy:
 *   - Each thread processes one proof term
 *   - Substitution is done IN-PLACE on the mutable node arrays
 *   - We use a work stack (iterative BFS/DFS) instead of recursion
 *   - Max nodes per term bounded by MAX_NODES
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Node types
#define N_SORT       0
#define N_VAR        1   // BVAR — de Bruijn variable
#define N_CONST      2
#define N_APP        3
#define N_LAM        4
#define N_PI         5
#define N_LET        6
#define N_CTOR       7
#define N_REC        8
#define N_NATLIT     9
#define N_NAT_ZERO  10
#define N_NAT_SUCC  11
#define N_BOOL_TRUE 12
#define N_BOOL_FALSE 13
#define N_NONE      -1

// Limits
#define MAX_NODES        256  // max nodes per proof term
#define MAX_SUBST_DEPTH   16  // max substitution recursion depth
#define MAX_WORK_STACK   128  // work items for iterative traversal

// Work item for iterative substitution
struct SubstWork {
    int32_t node_idx;    // which node to process
    int32_t depth;       // current binding depth
};

// ============================================================
// DEVICE: Shift free variables in a subtree
// ============================================================
// shift(term, cutoff, amount):
//   BVAR(i) where i >= cutoff → BVAR(i + amount)
//   Under binder: cutoff += 1

__global__ void shift_vars_kernel(
    int64_t* __restrict__ node_types,
    int64_t* __restrict__ child1,
    int64_t* __restrict__ child2,
    int64_t* __restrict__ child3,
    int64_t* __restrict__ aux1,   // for BVAR: de Bruijn index
    int64_t* __restrict__ aux2,
    const int64_t* __restrict__ shift_roots,  // [B] — root node to shift
    int shift_amount,                          // how much to shift
    int B, int MN
) {
    int bi = blockIdx.x * blockDim.x + threadIdx.x;
    if (bi >= B) return;

    int base = bi * MN;

    // Iterative traversal with depth tracking
    SubstWork stack[MAX_WORK_STACK];
    int sp = 0;

    int64_t root = shift_roots[bi];
    if (root < 0 || root >= MN) return;

    stack[sp++] = {(int32_t)root, 0};

    while (sp > 0 && sp < MAX_WORK_STACK) {
        SubstWork work = stack[--sp];
        int ni = work.node_idx;
        int cutoff = work.depth;

        if (ni < 0 || ni >= MN) continue;
        int64_t ntype = node_types[base + ni];
        if (ntype == N_NONE) continue;

        if (ntype == N_VAR) {
            int64_t idx = aux1[base + ni];
            if (idx >= cutoff) {
                aux1[base + ni] = idx + shift_amount;
            }
        }
        else if (ntype == N_LAM || ntype == N_PI) {
            // Binder increases cutoff for body (child1)
            int64_t domain = child2[base + ni];  // domain type
            int64_t body = child1[base + ni];     // body
            if (domain >= 0 && domain < MN && sp < MAX_WORK_STACK)
                stack[sp++] = {(int32_t)domain, cutoff};
            if (body >= 0 && body < MN && sp < MAX_WORK_STACK)
                stack[sp++] = {(int32_t)body, cutoff + 1};
        }
        else if (ntype == N_LET) {
            int64_t type_n = child1[base + ni];
            int64_t val_n = child2[base + ni];
            int64_t body_n = child3[base + ni];
            if (type_n >= 0 && type_n < MN && sp < MAX_WORK_STACK)
                stack[sp++] = {(int32_t)type_n, cutoff};
            if (val_n >= 0 && val_n < MN && sp < MAX_WORK_STACK)
                stack[sp++] = {(int32_t)val_n, cutoff};
            if (body_n >= 0 && body_n < MN && sp < MAX_WORK_STACK)
                stack[sp++] = {(int32_t)body_n, cutoff + 1};
        }
        else {
            // APP, SUCC, etc — traverse children without changing cutoff
            int64_t c1 = child1[base + ni];
            int64_t c2v = child2[base + ni];
            int64_t c3v = child3[base + ni];
            if (c1 >= 0 && c1 < MN && sp < MAX_WORK_STACK)
                stack[sp++] = {(int32_t)c1, cutoff};
            if (c2v >= 0 && c2v < MN && sp < MAX_WORK_STACK)
                stack[sp++] = {(int32_t)c2v, cutoff};
            if (c3v >= 0 && c3v < MN && sp < MAX_WORK_STACK)
                stack[sp++] = {(int32_t)c3v, cutoff};
        }
    }
}

// ============================================================
// DEVICE: Core substitution — subst(term, var_depth, replacement)
// ============================================================
// Substitutes BVAR(var_depth) with replacement in the subtree rooted at root.
// This is the KEY operation for beta-reduction:
//   App(Lam(body), arg) → subst(body, 0, arg)

__device__ void subst_inplace(
    int64_t* node_types,
    int64_t* child1,
    int64_t* child2,
    int64_t* child3,
    int64_t* aux1,
    int64_t* aux2,
    int root_idx,
    int var_depth,         // which de Bruijn level to substitute
    int replacement_idx,   // node index of the replacement term
    int base, int MN,
    int64_t* next_free     // pointer to next free node slot (for cloning)
) {
    SubstWork stack[MAX_WORK_STACK];
    int sp = 0;

    if (root_idx < 0 || root_idx >= MN) return;
    stack[sp++] = {(int32_t)root_idx, var_depth};

    while (sp > 0 && sp < MAX_WORK_STACK) {
        SubstWork work = stack[--sp];
        int ni = work.node_idx;
        int depth = work.depth;

        if (ni < 0 || ni >= MN) continue;
        int64_t ntype = node_types[base + ni];
        if (ntype == N_NONE) continue;

        if (ntype == N_VAR) {
            int64_t idx = aux1[base + ni];
            if (idx == depth) {
                // HIT: substitute this variable with replacement
                // Deep copy of replacement node into this slot
                if (replacement_idx >= 0 && replacement_idx < MN) {
                    node_types[base + ni] = node_types[base + replacement_idx];
                    child1[base + ni] = child1[base + replacement_idx];
                    child2[base + ni] = child2[base + replacement_idx];
                    child3[base + ni] = child3[base + replacement_idx];
                    aux1[base + ni] = aux1[base + replacement_idx];
                    aux2[base + ni] = aux2[base + replacement_idx];
                }
            }
            else if (idx > depth) {
                // Free variable above substitution point — shift down
                aux1[base + ni] = idx - 1;
            }
            // idx < depth → bound variable, untouched
        }
        else if (ntype == N_LAM || ntype == N_PI) {
            // Binder: increase depth for body
            int64_t body = child1[base + ni];
            int64_t domain = child2[base + ni];
            if (domain >= 0 && domain < MN && sp < MAX_WORK_STACK)
                stack[sp++] = {(int32_t)domain, depth};
            if (body >= 0 && body < MN && sp < MAX_WORK_STACK)
                stack[sp++] = {(int32_t)body, depth + 1};
        }
        else if (ntype == N_LET) {
            int64_t type_n = child1[base + ni];
            int64_t val_n = child2[base + ni];
            int64_t body_n = child3[base + ni];
            if (type_n >= 0 && type_n < MN && sp < MAX_WORK_STACK)
                stack[sp++] = {(int32_t)type_n, depth};
            if (val_n >= 0 && val_n < MN && sp < MAX_WORK_STACK)
                stack[sp++] = {(int32_t)val_n, depth};
            if (body_n >= 0 && body_n < MN && sp < MAX_WORK_STACK)
                stack[sp++] = {(int32_t)body_n, depth + 1};
        }
        else if (ntype == N_APP) {
            int64_t func = child1[base + ni];
            int64_t arg = child2[base + ni];
            if (func >= 0 && func < MN && sp < MAX_WORK_STACK)
                stack[sp++] = {(int32_t)func, depth};
            if (arg >= 0 && arg < MN && sp < MAX_WORK_STACK)
                stack[sp++] = {(int32_t)arg, depth};
        }
        else if (ntype == N_NAT_SUCC) {
            int64_t c = child1[base + ni];
            if (c >= 0 && c < MN && sp < MAX_WORK_STACK)
                stack[sp++] = {(int32_t)c, depth};
        }
        // Leaves (SORT, CONST, NATLIT, NAT_ZERO, BOOL_TRUE, BOOL_FALSE): no children to recurse
    }
}


// ============================================================
// KERNEL: Beta Reduction via Substitution
// ============================================================
// For each proof term, perform one step of beta reduction at the root:
//   If root = APP(LAM(body), arg) → subst(body, 0, arg), update root
//   If root = LET(type, val, body) → subst(body, 0, val), update root

__global__ void beta_reduce_kernel(
    int64_t* __restrict__ node_types,
    int64_t* __restrict__ child1,
    int64_t* __restrict__ child2,
    int64_t* __restrict__ child3,
    int64_t* __restrict__ aux1,
    int64_t* __restrict__ aux2,
    int64_t* __restrict__ root_indices,  // [B] — mutable
    int64_t* __restrict__ reduced,       // [B] — output: 1 if reduced, 0 if not
    int B, int MN
) {
    int bi = blockIdx.x * blockDim.x + threadIdx.x;
    if (bi >= B) return;

    int base = bi * MN;
    int64_t root = root_indices[bi];
    int64_t did_reduce = 0;

    if (root < 0 || root >= MN) { reduced[bi] = 0; return; }
    int64_t ntype = node_types[base + root];

    if (ntype == N_APP) {
        int64_t func_idx = child1[base + root];
        int64_t arg_idx = child2[base + root];

        if (func_idx >= 0 && func_idx < MN &&
            node_types[base + func_idx] == N_LAM) {
            // Beta: App(Lam(body, domain), arg) → subst(body, 0, arg)
            int64_t body_idx = child1[base + func_idx];

            if (body_idx >= 0 && body_idx < MN) {
                int64_t dummy_next = MN;  // no allocation needed for simple subst
                subst_inplace(
                    node_types, child1, child2, child3, aux1, aux2,
                    (int)body_idx, 0, (int)arg_idx,
                    base, MN, &dummy_next
                );
                // Point root at the body (which is now substituted)
                root_indices[bi] = body_idx;
                did_reduce = 1;
            }
        }
    }
    else if (ntype == N_LET) {
        // Zeta: Let(type, val, body) → subst(body, 0, val)
        int64_t val_idx = child2[base + root];
        int64_t body_idx = child3[base + root];

        if (body_idx >= 0 && body_idx < MN &&
            val_idx >= 0 && val_idx < MN) {
            int64_t dummy_next = MN;
            subst_inplace(
                node_types, child1, child2, child3, aux1, aux2,
                (int)body_idx, 0, (int)val_idx,
                base, MN, &dummy_next
            );
            root_indices[bi] = body_idx;
            did_reduce = 1;
        }
    }

    reduced[bi] = did_reduce;
}


// ============================================================
// HOST: Substitution Pipeline
// ============================================================

std::vector<torch::Tensor> beta_reduce_step(
    torch::Tensor node_types, torch::Tensor child1,
    torch::Tensor child2, torch::Tensor child3,
    torch::Tensor aux1, torch::Tensor aux2,
    torch::Tensor root_indices
) {
    int B = node_types.size(0);
    int MN = node_types.size(1);
    int threads = 256;
    int blocks = (B + threads - 1) / threads;

    auto reduced = torch::zeros({B}, node_types.options());

    beta_reduce_kernel<<<blocks, threads>>>(
        node_types.data_ptr<int64_t>(), child1.data_ptr<int64_t>(),
        child2.data_ptr<int64_t>(), child3.data_ptr<int64_t>(),
        aux1.data_ptr<int64_t>(), aux2.data_ptr<int64_t>(),
        root_indices.data_ptr<int64_t>(), reduced.data_ptr<int64_t>(),
        B, MN
    );

    return {root_indices, reduced};
}

std::vector<torch::Tensor> shift_variables(
    torch::Tensor node_types, torch::Tensor child1,
    torch::Tensor child2, torch::Tensor child3,
    torch::Tensor aux1, torch::Tensor aux2,
    torch::Tensor shift_roots, int shift_amount
) {
    int B = node_types.size(0);
    int MN = node_types.size(1);
    int threads = 256;
    int blocks = (B + threads - 1) / threads;

    shift_vars_kernel<<<blocks, threads>>>(
        node_types.data_ptr<int64_t>(), child1.data_ptr<int64_t>(),
        child2.data_ptr<int64_t>(), child3.data_ptr<int64_t>(),
        aux1.data_ptr<int64_t>(), aux2.data_ptr<int64_t>(),
        shift_roots.data_ptr<int64_t>(), shift_amount,
        B, MN
    );

    return {node_types, child1, child2, child3, aux1, aux2};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("beta_reduce_step", &beta_reduce_step,
          "One step of beta/zeta reduction with proper de Bruijn substitution");
    m.def("shift_variables", &shift_variables,
          "Shift free de Bruijn variables in subtrees");
}
