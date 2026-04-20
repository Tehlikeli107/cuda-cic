/*
 * CIC Unified Engine: Single-Pass WHNF + TypeCheck + DefEq
 * ==========================================================
 * Replaces the 4 separate kernels with one unified pipeline.
 *
 * Each CUDA thread handles one proof term end-to-end:
 *   1. WHNF reduction with proper de Bruijn substitution
 *   2. Level-by-level type inference
 *   3. Result extraction
 *
 * Key improvement over v1: uses cic_subst.cu's substitution engine
 * for correct beta-reduction instead of the simplified in-place copy.
 *
 * Node encoding (same as before):
 *   [node_type, child1, child2, child3, aux1, aux2, level]
 *
 * For BVAR: aux1 = de Bruijn index, aux2 = resolved type hint
 * For CONST: aux1 = constant ID
 * For LAM/PI: child1 = body, child2 = domain type
 * For APP: child1 = function, child2 = argument
 * For LET: child1 = type annotation, child2 = value, child3 = body
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ============================================================
// NODE TYPES
// ============================================================
#define N_SORT       0
#define N_VAR        1   // de Bruijn variable
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
#define N_STRLIT    14
#define N_NONE      -1

// Type constants
#define T_ERROR   0
#define T_PROP    1
#define T_TYPE    2
#define T_TYPE1   3
#define T_NAT     10
#define T_BOOL    11
#define T_STRING  12
#define T_LIST    13

// Hash constants
#define PRIME1      1000003LL
#define PRIME2      999983LL
#define PI_SALT     0x50000000LL
#define SORT_SALT   0x60000000LL
#define HASH_MOD    1048576LL
#define TABLE_SIZE  8388708LL

// Limits
#define MAX_WHNF_STEPS   32   // max reduction steps per term
#define MAX_CONSTS    65536
#define MAX_WORK_STACK  64    // substitution work stack

// ============================================================
// DEVICE: Type hashing functions
// ============================================================

__device__ inline int64_t sort_hash(int64_t level) {
    if (level == 0) return T_PROP;
    if (level == 1) return T_TYPE;
    if (level == 2) return T_TYPE1;
    return ((level * PRIME1 + SORT_SALT) % HASH_MOD) + HASH_MOD;
}

__device__ inline int64_t pi_hash(int64_t dom, int64_t cod) {
    return ((dom * PRIME1 + cod * PRIME2 + PI_SALT) % HASH_MOD) + 2*HASH_MOD;
}

__device__ inline int64_t sort_of_sort(int64_t level) {
    return sort_hash(level + 1);
}

__device__ inline int64_t imax_level(int64_t u, int64_t v) {
    // imax(u, 0) = 0, imax(u, v) = max(u, v) otherwise
    if (v == 0) return 0;
    return (u > v) ? u : v;
}


// ============================================================
// DEVICE: Out-of-place Substitution (Allocating new nodes)
// Used for Dependent Type Checking (e.g., Pi(x:A, B) a -> B[x:=a])
// Recursive traversal simulating Deep Copy with Hash-Consing
// ============================================================

__device__ void subst_alloc_dev(
    int64_t* node_types, int64_t* child1, int64_t* child2,
    int64_t* child3, int64_t* aux1, int64_t* aux2, int64_t* levels,
    int root_idx, int var_depth, int replacement_idx,
    int base, int MN, int* pool_ptr, int64_t* out_idx
) {
    if (root_idx < 0 || root_idx >= MN) {
        *out_idx = root_idx;
        return;
    }

    int64_t ntype = node_types[base + root_idx];

    // If it's the exact bound variable we want to substitute
    if (ntype == N_VAR && aux1[base + root_idx] == var_depth) {
        *out_idx = replacement_idx;
        return;
    }
    
    // If it's a bound variable but a deeper one, we might need to shift it down 
    // (de Bruijn shifting), but typically for simple substitution `B[x:=a]` we just leave it alone
    // or decrement if `B` was under a binder that we just removed.
    // For pure Types as Terms instantiation we ignore shifting in this simplified Phase 8 step.
    if (ntype == N_VAR || ntype == N_CONST || ntype == N_SORT || ntype == N_NATLIT ||
        ntype == N_NAT_ZERO || ntype == N_BOOL_TRUE || ntype == N_BOOL_FALSE || ntype == N_STRLIT) {
        *out_idx = root_idx; // Leaf nodes: just point to them (Structural Sharing)
        return;
    }

    // It's a compound node. We must recursively substitute into its children.
    // GPU limitation: No true recursion. We should ideally use an explicit stack, 
    // but for simplicity in types (which are usually small ASTs), we can simulate.
    // To avoid stack overflows in CUDA, we do a bounded iterative copy or limited recursion.
    // For Phase 8.3 we use a macro-like iterative approach.

    int64_t c1 = child1[base + root_idx];
    int64_t c2 = child2[base + root_idx];
    int64_t c3 = child3[base + root_idx];
    
    int64_t new_c1 = c1;
    int64_t new_c2 = c2;
    int64_t new_c3 = c3;

    int new_depth = var_depth + ((ntype == N_LAM || ntype == N_PI || ntype == N_LET) ? 1 : 0);

    // Extremely shallow recursive call (CUDA compiler usually inlines this if depth is small)
    // In production, this MUST be flattened to a WorkStack like in WHNF.
    if (c1 >= 0 && c1 < MN) {
        subst_alloc_dev(node_types, child1, child2, child3, aux1, aux2, levels,
                        c1, new_depth, replacement_idx, base, MN, pool_ptr, &new_c1);
    }
    if (c2 >= 0 && c2 < MN) {
        subst_alloc_dev(node_types, child1, child2, child3, aux1, aux2, levels,
                        c2, (ntype == N_LET ? new_depth : var_depth), replacement_idx, base, MN, pool_ptr, &new_c2);
    }
    if (c3 >= 0 && c3 < MN) {
        subst_alloc_dev(node_types, child1, child2, child3, aux1, aux2, levels,
                        c3, new_depth, replacement_idx, base, MN, pool_ptr, &new_c3);
    }

    // If nothing changed, return original (Structural Sharing)
    if (new_c1 == c1 && new_c2 == c2 && new_c3 == c3) {
        *out_idx = root_idx;
        return;
    }

    // Something changed. Allocate a new node via Hash-Consing.
    int found = -1;
    for (int i = MN - 1; i > *pool_ptr; i--) {
        if (node_types[base + i] == ntype &&
            child1[base + i] == new_c1 &&
            child2[base + i] == new_c2 &&
            child3[base + i] == new_c3 &&
            aux1[base + i] == aux1[base + root_idx] &&
            aux2[base + i] == aux2[base + root_idx] &&
            levels[base + i] == levels[base + root_idx]) {
            found = i;
            break;
        }
    }

    if (found != -1) {
        *out_idx = found;
    } else if (*pool_ptr >= 0) {
        int idx = *pool_ptr;
        node_types[base + idx] = ntype;
        child1[base + idx] = new_c1;
        child2[base + idx] = new_c2;
        child3[base + idx] = new_c3;
        aux1[base + idx] = aux1[base + root_idx];
        aux2[base + idx] = aux2[base + root_idx];
        levels[base + idx] = levels[base + root_idx];
        (*pool_ptr)--;
        *out_idx = idx;
    } else {
        *out_idx = -1; // Out of Memory
    }
}

// ============================================================
// DEVICE: In-place de Bruijn substitution (from cic_subst.cu)
// ============================================================

struct WorkItem {
    int32_t node_idx;
    int32_t depth;
};

__device__ void subst_inplace_dev(
    int64_t* node_types, int64_t* child1, int64_t* child2,
    int64_t* child3, int64_t* aux1, int64_t* aux2,
    int root_idx, int var_depth, int replacement_idx,
    int base, int MN
) {
    WorkItem stack[MAX_WORK_STACK];
    int sp = 0;

    if (root_idx < 0 || root_idx >= MN) return;
    stack[sp++] = {(int32_t)root_idx, var_depth};

    while (sp > 0 && sp < MAX_WORK_STACK) {
        WorkItem work = stack[--sp];
        int ni = work.node_idx;
        int depth = work.depth;

        if (ni < 0 || ni >= MN) continue;
        int64_t ntype = node_types[base + ni];
        if (ntype == N_NONE) continue;

        if (ntype == N_VAR) {
            int64_t idx = aux1[base + ni];
            if (idx == depth && replacement_idx >= 0 && replacement_idx < MN) {
                // Substitute: copy replacement node
                node_types[base + ni] = node_types[base + replacement_idx];
                child1[base + ni] = child1[base + replacement_idx];
                child2[base + ni] = child2[base + replacement_idx];
                child3[base + ni] = child3[base + replacement_idx];
                aux1[base + ni] = aux1[base + replacement_idx];
                aux2[base + ni] = aux2[base + replacement_idx];
            }
            else if (idx > depth) {
                aux1[base + ni] = idx - 1;  // shift down free variable
            }
        }
        else if (ntype == N_LAM || ntype == N_PI) {
            int64_t body = child1[base + ni];
            int64_t domain = child2[base + ni];
            if (domain >= 0 && domain < MN && sp < MAX_WORK_STACK)
                stack[sp++] = {(int32_t)domain, depth};
            if (body >= 0 && body < MN && sp < MAX_WORK_STACK)
                stack[sp++] = {(int32_t)body, depth + 1};
        }
        else if (ntype == N_LET) {
            int64_t t = child1[base + ni], v = child2[base + ni], b = child3[base + ni];
            if (t >= 0 && t < MN && sp < MAX_WORK_STACK) stack[sp++] = {(int32_t)t, depth};
            if (v >= 0 && v < MN && sp < MAX_WORK_STACK) stack[sp++] = {(int32_t)v, depth};
            if (b >= 0 && b < MN && sp < MAX_WORK_STACK) stack[sp++] = {(int32_t)b, depth + 1};
        }
        else if (ntype == N_APP) {
            int64_t f = child1[base + ni], a = child2[base + ni];
            if (f >= 0 && f < MN && sp < MAX_WORK_STACK) stack[sp++] = {(int32_t)f, depth};
            if (a >= 0 && a < MN && sp < MAX_WORK_STACK) stack[sp++] = {(int32_t)a, depth};
        }
        else if (ntype == N_NAT_SUCC) {
            int64_t c = child1[base + ni];
            if (c >= 0 && c < MN && sp < MAX_WORK_STACK) stack[sp++] = {(int32_t)c, depth};
        }
    }
}


// ============================================================
// KERNEL: Unified WHNF Reduction (with proper substitution)
// ============================================================

__global__ void engine_whnf_kernel(
    int64_t* __restrict__ node_types,
    int64_t* __restrict__ child1,
    int64_t* __restrict__ child2,
    int64_t* __restrict__ child3,
    int64_t* __restrict__ aux1,
    int64_t* __restrict__ aux2,
    int64_t* __restrict__ levels,
    int64_t* __restrict__ root_indices,
    const int64_t* __restrict__ def_types,    // [MAX_CONSTS] — definition result types
    int64_t* __restrict__ whnf_steps,
    int B, int MN
) {
    int bi = blockIdx.x * blockDim.x + threadIdx.x;
    if (bi >= B) return;

    int base = bi * MN;
    int64_t root = root_indices[bi];
    int steps = 0;

    // --- Thread-Local Allocator (IOTA/Pattern Matching) ---
    // En son node nerede bitiyor bulmamız lazım. Genelde MN'in yarısına kadarı Lean'den gelen ağaçla doludur.
    // Şimdilik hızlıca son node'u bulmak yerine (ki O(MN) sürer) allocate etmek için "tersten"
    // (MN - 1'den aşağıya doğru) bir pool kullanacağız.
    int pool_ptr = MN - 1;

    // Macro for fast thread-local allocation with Linear-Scan Hash-Consing (Deduplication)
    #define ALLOC_NODE_HASH_CONS(kind, c1, c2, c3, a1, a2, lvl, out_idx) do { \
        int found = -1; \
        for (int i = MN - 1; i > pool_ptr; i--) { \
            if (node_types[base + i] == (kind) && \
                child1[base + i] == (c1) && \
                child2[base + i] == (c2) && \
                child3[base + i] == (c3) && \
                aux1[base + i] == (a1) && \
                aux2[base + i] == (a2) && \
                levels[base + i] == (lvl)) { \
                found = i; \
                break; \
            } \
        } \
        if (found != -1) { \
            (out_idx) = found; \
        } else if (pool_ptr >= 0) { \
            node_types[base + pool_ptr] = (kind); \
            child1[base + pool_ptr] = (c1); \
            child2[base + pool_ptr] = (c2); \
            child3[base + pool_ptr] = (c3); \
            aux1[base + pool_ptr] = (a1); \
            aux2[base + pool_ptr] = (a2); \
            levels[base + pool_ptr] = (lvl); \
            (out_idx) = pool_ptr; \
            pool_ptr--; \
        } else { \
            (out_idx) = -1; \
        } \
    } while(0)

    for (int step = 0; step < MAX_WHNF_STEPS; step++) {
        if (root < 0 || root >= MN) break;
        int64_t ntype = node_types[base + root];

        if (ntype == N_APP) {
            int64_t func_idx = child1[base + root];
            int64_t arg_idx = child2[base + root];

            if (func_idx >= 0 && func_idx < MN &&
                node_types[base + func_idx] == N_LAM) {
                // BETA: App(Lam(body, domain), arg) → subst(body, 0, arg)
                int64_t body_idx = child1[base + func_idx];

                if (body_idx >= 0 && body_idx < MN) {
                    // Proper de Bruijn substitution
                    subst_inplace_dev(
                        node_types, child1, child2, child3, aux1, aux2,
                        (int)body_idx, 0, (int)arg_idx, base, MN
                    );
                    root = body_idx;
                    steps++;
                    continue;
                }
            }
            
            // IOTA (Recursor): App(App(...App(Rec, motive), minor), major)
            // If the head is a recursor and the major premise is a constructor we should reduce.
            int spine[32];
            int spine_idx = 0;
            int head_idx = root;
            
            while (head_idx >= 0 && head_idx < MN && node_types[base + head_idx] == N_APP && spine_idx < 32) {
                spine[spine_idx++] = child2[base + head_idx];
                head_idx = child1[base + head_idx];
            }
            
            if (head_idx >= 0 && head_idx < MN && node_types[base + head_idx] == N_REC) {
                // head is N_REC. The last element in spine is the major premise.
                int major_premise_idx = spine[0]; // spine[0] is the outermost argument
                if (major_premise_idx >= 0 && major_premise_idx < MN) {
                    int major_type = node_types[base + major_premise_idx];
                    if (major_type == N_CTOR || major_type == N_NAT_ZERO || major_type == N_NAT_SUCC || major_type == N_BOOL_TRUE || major_type == N_BOOL_FALSE) {
                        // TODO: Generic rule-based lookup
                        // Phase 7.2: For now, implement fast path for known recursors
                        int64_t rec_id = aux1[base + head_idx];
                        
                        // Let's assume we can map rec_id directly or we implement Nat and Bool first to test the alloc
                        if (major_type == N_NAT_ZERO && spine_idx >= 2) {
                            // Nat.rec motive base step Nat.zero -> base (which is spine[1])
                            root = spine[1];
                            steps++;
                            continue;
                        }
                        else if (major_type == N_BOOL_TRUE && spine_idx >= 3) {
                            // Bool.rec motive t_case f_case true -> t_case
                            root = spine[2]; // the true case
                            steps++;
                            continue;
                        }
                        else if (major_type == N_BOOL_FALSE && spine_idx >= 3) {
                            // Bool.rec motive t_case f_case false -> f_case
                            root = spine[1]; // the false case
                            steps++;
                            continue;
                        }
                        else if (major_type == N_NAT_SUCC && spine_idx >= 3) {
                            // spine[0] = (Nat.succ n)
                            // spine[1] = step
                            // spine[2] = base
                            // spine[3] = motive (if present)
                            
                            int64_t n_idx = child1[base + major_premise_idx];
                            int64_t step_idx = spine[1];
                            
                            // Rebuild (Nat.rec motive base step) which is just the head applied to motive, base, step
                            // We can find this intermediate application node in the tree natively:
                            // App( App( App(Rec, motive), base ), step )
                            // We know this is exactly the parent of the major premise application.
                            // In spine traversal, `spine_idx` counts arguments from outside in.
                            // root is `App(App_step, major_premise)`
                            int64_t app_step_idx = child1[base + root];
                            
                            // 1. Allocate: App(app_step_idx, n_idx)  == (Nat.rec motive base step n)
                            int64_t rec_call_idx;
                            ALLOC_NODE_HASH_CONS(N_APP, app_step_idx, n_idx, -1, 0, 0, 0, rec_call_idx);
                            
                            // 2. Allocate: App(step_idx, n_idx) == step n
                            int64_t step_n_idx;
                            ALLOC_NODE_HASH_CONS(N_APP, step_idx, n_idx, -1, 0, 0, 0, step_n_idx);
                            
                            // 3. Allocate: App(step_n_idx, rec_call_idx) == step n (Nat.rec motive base step n)
                            int64_t final_app_idx;
                            ALLOC_NODE_HASH_CONS(N_APP, step_n_idx, rec_call_idx, -1, 0, 0, 0, final_app_idx);
                            
                            if (final_app_idx >= 0) {
                                root = final_app_idx;
                                steps++;
                                continue;
                            }
                        }
                    }
                }
            }

            // DELTA through APP: App(Const(f), arg) where f has definition
            if (func_idx >= 0 && func_idx < MN &&
                node_types[base + func_idx] == N_CONST) {
                int64_t cid = aux1[base + func_idx];
                if (cid >= 0 && cid < MAX_CONSTS && def_types[cid] != 0) {
                    aux1[base + func_idx] = def_types[cid];
                    node_types[base + func_idx] = N_VAR;
                    steps++;
                    continue;
                }
            }
            break;
        }
        else if (ntype == N_LET) {
            // ZETA: Let(type, val, body) → subst(body, 0, val)
            int64_t val_idx = child2[base + root];
            int64_t body_idx = child3[base + root];

            if (body_idx >= 0 && body_idx < MN &&
                val_idx >= 0 && val_idx < MN) {
                subst_inplace_dev(
                    node_types, child1, child2, child3, aux1, aux2,
                    (int)body_idx, 0, (int)val_idx, base, MN
                );
                root = body_idx;
                steps++;
                continue;
            }
            break;
        }
        else if (ntype == N_CONST) {
            // DELTA: unfold constant definition
            int64_t cid = aux1[base + root];
            if (cid >= 0 && cid < MAX_CONSTS && def_types[cid] != 0) {
                aux1[base + root] = def_types[cid];
                node_types[base + root] = N_VAR;
                steps++;
                continue;
            }
            break;
        }
        else if (ntype == N_NAT_SUCC) {
            // NatLit computation: S(n) → n+1
            int64_t c = child1[base + root];
            if (c >= 0 && c < MN && node_types[base + c] == N_NATLIT) {
                node_types[base + root] = N_NATLIT;
                aux1[base + root] = aux1[base + c] + 1;
                child1[base + root] = -1;
                steps++;
                continue;
            }
            break;
        }
        else {
            break;  // WHNF: Sort, Var, Lam, Pi, Ctor, NatLit are head-normal
        }
    }

    root_indices[bi] = root;
    whnf_steps[bi] = steps;
}


// ============================================================
// KERNEL: Unified Type Checking (Term-Parallel)
// ============================================================

__global__ void engine_type_check_kernel(
    int64_t* __restrict__ node_types,
    int64_t* __restrict__ child1,
    int64_t* __restrict__ child2,
    int64_t* __restrict__ child3,
    int64_t* __restrict__ aux1,
    int64_t* __restrict__ aux2,
    int64_t* __restrict__ levels,
    int64_t* __restrict__ result,
    int64_t* __restrict__ pi_lookup,
    const int64_t* __restrict__ const_types,
    int target_level, int B, int MN
) {
    // Term-Parallel execution: One thread per Proof Term (bi)
    int bi = blockIdx.x * blockDim.x + threadIdx.x;
    if (bi >= B) return;

    int base = bi * MN;

    // Track the end of the term for local allocations (e.g. Dependent Pattern Matching substitution)
    int pool_ptr = MN - 1;

    // Process ALL nodes of the current target_level belonging to this proof term (bi)
    for (int ni = 0; ni < MN; ni++) {
        if (levels[base + ni] != target_level) continue;

        int64_t ntype = node_types[base + ni];
        if (ntype == N_NONE) continue;

        int64_t c1 = child1[base + ni], c2 = child2[base + ni], c3 = child3[base + ni];
        int64_t a1 = aux1[base + ni], a2 = aux2[base + ni];

        int64_t ct1 = T_ERROR, ct2 = T_ERROR, ct3 = T_ERROR;
        if (c1 >= 0 && c1 < MN) ct1 = result[base + c1];
        if (c2 >= 0 && c2 < MN) ct2 = result[base + c2];
        if (c3 >= 0 && c3 < MN) ct3 = result[base + c3];

        int64_t res = T_ERROR;

        switch (ntype) {
            case N_SORT:
                res = sort_of_sort(a1);
                break;

            case N_VAR:
                res = (a2 != 0) ? a2 : a1;
                break;

            case N_CONST:
                if (a1 >= 0 && a1 < MAX_CONSTS)
                    res = const_types[a1];
                break;

            case N_NATLIT:
            case N_NAT_ZERO:
                res = T_NAT;
                break;

            case N_NAT_SUCC:
                res = (ct1 == T_NAT) ? T_NAT : T_ERROR;
                break;

            case N_BOOL_TRUE:
            case N_BOOL_FALSE:
                res = T_BOOL;
                break;

            case N_STRLIT:
                res = T_STRING;
                break;

            case N_LAM: {
                int64_t body_type = ct1;
                int64_t dom_type = ct2;

                int64_t dom = (dom_type == T_TYPE && c2 >= 0 && c2 < MN) ?
                              const_types[aux1[base + c2]] : dom_type;
                if (c2 >= 0 && c2 < MN && node_types[base + c2] == N_CONST) {
                    int64_t cid = aux1[base + c2];
                    if (cid >= 0 && cid < MAX_CONSTS) {
                        int64_t ct = const_types[cid];
                        if (ct == T_TYPE) {
                            if (cid == 0) dom = T_NAT;       // Nat
                            else if (cid == 1) dom = T_BOOL;  // Bool
                            else dom = ct;
                        }
                    }
                }

                if (dom != T_ERROR && body_type != T_ERROR) {
                    int64_t h = pi_hash(dom, body_type);
                    if (h >= 0 && h < TABLE_SIZE) {
                        // Safe to use non-atomic since thread owns its pi_lookup portion, 
                        // but since pi_lookup is global across all terms right now we keep atomicExch.
                        atomicExch((unsigned long long*)&pi_lookup[h*2],
                                  (unsigned long long)dom);
                        atomicExch((unsigned long long*)&pi_lookup[h*2+1],
                                  (unsigned long long)body_type);
                    }
                    res = h;
                }
                break;
            }

            case N_PI: {
                int64_t l1 = a1, l2 = a2;
                res = sort_hash(imax_level(l1, l2));
                break;
            }

            case N_APP: {
                int64_t func_type = ct1;
                int64_t arg_type = ct2;

                if (func_type > 0 && func_type < TABLE_SIZE) {
                    int64_t dom = pi_lookup[func_type * 2];
                    int64_t cod = pi_lookup[func_type * 2 + 1];
                    if (dom != 0 && dom == arg_type) {
                        // Phase 8.3 TODO: Return a new AST node created by substitution (cod[x:=arg])
                        // For fully dependent types (Types as Terms), `cod` should be an AST node.
                        // Since `cod` here is currently a hash, we pass it through.
                        // In the future:
                        // int64_t new_cod;
                        // subst_alloc_dev(cod_ast_idx, 0, c2, &new_cod); 
                        // res = new_cod;
                        res = cod;
                    }
                }
                break;
            }

            case N_LET:
                res = ct3;
                break;

            case N_CTOR: {
                res = a2;
                break;
            }

            case N_REC: {
                if (ct1 != T_ERROR) res = ct1;
                break;
            }
        }

        result[base + ni] = res;
    }
}


// ============================================================
// KERNEL: Extract results
// ============================================================

__global__ void engine_extract_kernel(
    const int64_t* __restrict__ result,
    const int64_t* __restrict__ root_indices,
    int64_t* __restrict__ valid,
    int64_t* __restrict__ root_types,
    int B, int MN
) {
    int bi = blockIdx.x * blockDim.x + threadIdx.x;
    if (bi >= B) return;
    int64_t ri = root_indices[bi];
    int64_t rt = result[bi * MN + ri];
    valid[bi] = (rt != T_ERROR) ? 1 : 0;
    root_types[bi] = rt;
}


// ============================================================
// HOST: Unified Engine Pipeline
// ============================================================

std::vector<torch::Tensor> cic_engine_pipeline(
    torch::Tensor node_types, torch::Tensor child1, torch::Tensor child2,
    torch::Tensor child3, torch::Tensor aux1, torch::Tensor aux2,
    torch::Tensor levels, torch::Tensor root_indices,
    torch::Tensor pi_lookup, torch::Tensor const_types,
    torch::Tensor def_types, torch::Tensor ctor_tags, torch::Tensor rec_rules,
    int max_level
) {
    int B = node_types.size(0);
    int MN = node_types.size(1);

    auto result = torch::zeros({B, MN}, node_types.options());
    auto valid = torch::zeros({B}, node_types.options());
    auto root_types = torch::zeros({B}, node_types.options());
    auto whnf_steps = torch::zeros({B}, node_types.options());

    int threads = 256;

    // Phase 1: WHNF reduction with proper substitution
    int whnf_blocks = (B + threads - 1) / threads;
    engine_whnf_kernel<<<whnf_blocks, threads>>>(
        node_types.data_ptr<int64_t>(), child1.data_ptr<int64_t>(),
        child2.data_ptr<int64_t>(), child3.data_ptr<int64_t>(),
        aux1.data_ptr<int64_t>(), aux2.data_ptr<int64_t>(),
        levels.data_ptr<int64_t>(), root_indices.data_ptr<int64_t>(),
        def_types.data_ptr<int64_t>(), whnf_steps.data_ptr<int64_t>(),
        B, MN
    );

    // Phase 2: Type checking (level by level, Term-Parallel)
    // We launch 1 thread per proof term, not 1 thread per node.
    int tc_blocks = (B + threads - 1) / threads;
    for (int lv = 0; lv <= max_level; lv++) {
        engine_type_check_kernel<<<tc_blocks, threads>>>(
            node_types.data_ptr<int64_t>(), child1.data_ptr<int64_t>(),
            child2.data_ptr<int64_t>(), child3.data_ptr<int64_t>(),
            aux1.data_ptr<int64_t>(), aux2.data_ptr<int64_t>(),
            levels.data_ptr<int64_t>(), result.data_ptr<int64_t>(),
            pi_lookup.data_ptr<int64_t>(), const_types.data_ptr<int64_t>(),
            lv, B, MN
        );
    }

    // Phase 3: Extract results
    int rblocks = (B + threads - 1) / threads;
    engine_extract_kernel<<<rblocks, threads>>>(
        result.data_ptr<int64_t>(), root_indices.data_ptr<int64_t>(),
        valid.data_ptr<int64_t>(), root_types.data_ptr<int64_t>(),
        B, MN
    );

    return {valid, root_types, result, whnf_steps};
}


// Also expose backward-compatible cic_gpu_type_check
std::vector<torch::Tensor> cic_gpu_type_check(
    torch::Tensor node_types, torch::Tensor child1, torch::Tensor child2,
    torch::Tensor child3, torch::Tensor aux1, torch::Tensor aux2,
    torch::Tensor levels, torch::Tensor root_indices,
    torch::Tensor pi_lookup, torch::Tensor const_types,
    torch::Tensor def_types, torch::Tensor ctor_tags, torch::Tensor rec_rules,
    int max_level
) {
    auto results = cic_engine_pipeline(
        node_types, child1, child2, child3, aux1, aux2,
        levels, root_indices, pi_lookup, const_types,
        def_types, ctor_tags, rec_rules, max_level
    );
    return {results[0], results[1], results[2]};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cic_engine_pipeline", &cic_engine_pipeline,
          "Unified CIC engine: WHNF + TypeCheck with proper substitution");
    m.def("cic_gpu_type_check", &cic_gpu_type_check,
          "Backward-compatible type check interface");
}
