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
#define N_MVAR      15   // Metavariable (?m) for Higher-Order Unification
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

// Macro for fast thread-local allocation with Linear-Scan Hash-Consing (Deduplication)
#define ALLOC_NODE_HASH_CONS(kind, c1, c2, c3, a1, a2, lvl, out_idx) do { \
    int found = -1; \
    for (int i = MN - 1; i > *pool_ptr && i >= 0; i--) { \
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
    } else if (*pool_ptr >= 0 && *pool_ptr < MN) { \
        node_types[base + *pool_ptr] = (kind); \
        child1[base + *pool_ptr] = (c1); \
        child2[base + *pool_ptr] = (c2); \
        child3[base + *pool_ptr] = (c3); \
        aux1[base + *pool_ptr] = (a1); \
        aux2[base + *pool_ptr] = (a2); \
        levels[base + *pool_ptr] = (lvl); \
        (out_idx) = *pool_ptr; \
        (*pool_ptr)--; \
    } else { \
        (out_idx) = -1; \
    } \
} while(0)

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
);

__device__ void subst_inplace_dev(
    int64_t* node_types, int64_t* child1, int64_t* child2,
    int64_t* child3, int64_t* aux1, int64_t* aux2,
    int root_idx, int var_depth, int replacement_idx,
    int base, int MN
);

__device__ int whnf_single_dev(
    int64_t* node_types, int64_t* child1, int64_t* child2,
    int64_t* child3, int64_t* aux1, int64_t* aux2, int64_t* levels,
    const int64_t* def_types, const int64_t* ctor_tags, const int64_t* rec_rules,
    int root_idx, int base, int MN, int* pool_ptr, int* out_steps
) {
    int root = root_idx;
    int steps = 0;

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
            // If the head is a recursor and the major premise reduces to a constructor we should reduce.
            int spine[32];
            int spine_idx = 0;
            int head_idx = root;

            while (head_idx >= 0 && head_idx < MN && node_types[base + head_idx] == N_APP && spine_idx < 32) {
                spine[spine_idx++] = child2[base + head_idx];
                head_idx = child1[base + head_idx];
            }

            if (head_idx >= 0 && head_idx < MN && node_types[base + head_idx] == N_REC) {
                int major_premise_idx = spine[0];
                if (major_premise_idx >= 0 && major_premise_idx < MN) {

                    // Unroll constructor spine
                    int ctor_spine[16];
                    int ctor_spine_idx = 0;
                    int ctor_head = major_premise_idx;

                    // Assuming major premise is already in WHNF (starts with CTOR or APP of CTOR)
                    while (ctor_head >= 0 && ctor_head < MN && node_types[base + ctor_head] == N_APP && ctor_spine_idx < 16) {
                        ctor_spine[ctor_spine_idx++] = child2[base + ctor_head];
                        ctor_head = child1[base + ctor_head];
                    }

                    if (ctor_head >= 0 && ctor_head < MN && node_types[base + ctor_head] == N_CTOR) {
                        int64_t rec_id = aux1[base + head_idx];
                        int64_t rule = rec_rules[rec_id];
                        int n_params = (rule >> 16) & 0xFFFF;
                        int n_minors = rule & 0xFFFF;
                        int64_t ctor_tag = aux2[base + ctor_head];

                        if (spine_idx >= n_params + n_minors + 1) {
                            int minor_idx_in_spine = 1 + (n_minors - 1 - ctor_tag);
                            int64_t minor_premise_idx = spine[minor_idx_in_spine];

                            if (ctor_spine_idx == 0) {
                                // Constructor with NO arguments (e.g., Nat.zero) -> Just use the minor premise
                                root = minor_premise_idx;
                                steps++;
                                continue;
                            } else {
                                int64_t arg_x = ctor_spine[0];
                                int64_t rec_func_part = child1[base + root];

                                int64_t rec_call_idx;
                                ALLOC_NODE_HASH_CONS(N_APP, rec_func_part, arg_x, -1, 0, 0, 0, rec_call_idx);

                                int64_t step_n_idx;
                                ALLOC_NODE_HASH_CONS(N_APP, minor_premise_idx, arg_x, -1, 0, 0, 0, step_n_idx);

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
        else {
            break; // Already in WHNF
        }
    }
    
    if (out_steps) *out_steps += steps;
    return root;
}

// ============================================================
// DEVICE: Higher-Order Unification & Definitional Equality Engine
// The absolute core of Calculus of Inductive Constructions.
// Compares two ASTs for structural and definitional equality,
// reducing them to WHNF on-the-fly if needed. Supports Metavariables
// via Cooperative Shared Memory for active proof synthesis.
// ============================================================

__device__ bool unify_dev(
    int64_t* node_types, int64_t* child1, int64_t* child2,
    int64_t* child3, int64_t* aux1, int64_t* aux2, int64_t* levels,
    const int64_t* def_types, const int64_t* ctor_tags, const int64_t* rec_rules,
    int root_a, int root_b, int base, int MN, int* pool_ptr,
    int64_t* mvar_env, int max_mvars
);

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

// Macro for fast thread-local allocation with Linear-Scan Hash-Consing (Deduplication)
#define ALLOC_NODE_HASH_CONS(kind, c1, c2, c3, a1, a2, lvl, out_idx) do { \
    int found = -1; \
    for (int i = MN - 1; i > *pool_ptr && i >= 0; i--) { \
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
    } else if (*pool_ptr >= 0 && *pool_ptr < MN) { \
        node_types[base + *pool_ptr] = (kind); \
        child1[base + *pool_ptr] = (c1); \
        child2[base + *pool_ptr] = (c2); \
        child3[base + *pool_ptr] = (c3); \
        aux1[base + *pool_ptr] = (a1); \
        aux2[base + *pool_ptr] = (a2); \
        levels[base + *pool_ptr] = (lvl); \
        (out_idx) = *pool_ptr; \
        (*pool_ptr)--; \
    } else { \
        (out_idx) = -1; \
    } \
} while(0)

__device__ void subst_inplace_dev(
    int64_t* node_types, int64_t* child1, int64_t* child2,
    int64_t* child3, int64_t* aux1, int64_t* aux2,
    int root_idx, int var_depth, int replacement_idx,
    int base, int MN
);

__device__ void subst_alloc_dev(
    int64_t* node_types, int64_t* child1, int64_t* child2,
    int64_t* child3, int64_t* aux1, int64_t* aux2, int64_t* levels,
    int root_idx, int var_depth, int replacement_idx,
    int base, int MN, int* pool_ptr, int64_t* out_idx
);

__device__ int whnf_single_dev(
    int64_t* node_types, int64_t* child1, int64_t* child2,
    int64_t* child3, int64_t* aux1, int64_t* aux2, int64_t* levels,
    const int64_t* def_types, const int64_t* ctor_tags, const int64_t* rec_rules,
    int root_idx, int base, int MN, int* pool_ptr, int* out_steps
);

__device__ bool unify_dev(
    int64_t* node_types, int64_t* child1, int64_t* child2,
    int64_t* child3, int64_t* aux1, int64_t* aux2, int64_t* levels,
    const int64_t* def_types, const int64_t* ctor_tags, const int64_t* rec_rules,
    int root_a, int root_b, int base, int MN, int* pool_ptr,
    int64_t* mvar_env, int max_mvars
) {
    if (root_a == root_b) return true; // Fast Path: Hash-Consed Pointers

    if (root_a < 0 || root_a >= MN || root_b < 0 || root_b >= MN) return false;

    int64_t type_a = node_types[base + root_a];
    int64_t type_b = node_types[base + root_b];

    // Phase 8.6: Metavariable Assignment & Resolution
    if (type_a == N_MVAR && max_mvars > 0) {
        int64_t mvar_id = aux1[base + root_a] % max_mvars; 
        int64_t current_assignment = atomicCAS((unsigned long long*)&mvar_env[mvar_id], 0ULL, (unsigned long long)root_b);
        if (current_assignment == 0ULL) return true; // Successfully assigned ?m = root_b
        if (current_assignment == root_b) return true; // Already assigned to exactly the same node

        // If assigned to something else, we must unify the current assignment with root_b
        return unify_dev(node_types, child1, child2, child3, aux1, aux2, levels,
                         def_types, ctor_tags, rec_rules,
                         current_assignment, root_b, base, MN, pool_ptr, mvar_env, max_mvars);
    }

    if (type_b == N_MVAR && max_mvars > 0) {
        // Symmetric case
        int64_t mvar_id = aux1[base + root_b] % max_mvars;
        int64_t current_assignment = atomicCAS((unsigned long long*)&mvar_env[mvar_id], 0ULL, (unsigned long long)root_a);
        if (current_assignment == 0ULL) return true;
        if (current_assignment == root_a) return true;

        return unify_dev(node_types, child1, child2, child3, aux1, aux2, levels,
                         def_types, ctor_tags, rec_rules,
                         root_a, current_assignment, base, MN, pool_ptr, mvar_env, max_mvars);
    }

    // Structural Comparison
    if (type_a == type_b) {
        if (type_a == N_SORT || type_a == N_CONST || type_a == N_VAR || type_a == N_STRLIT ||
            type_a == N_NATLIT || type_a == N_NAT_ZERO || type_a == N_BOOL_TRUE || type_a == N_BOOL_FALSE) {
            return (aux1[base + root_a] == aux1[base + root_b]);
        }

        if (type_a == N_APP) {
            bool f_eq = unify_dev(node_types, child1, child2, child3, aux1, aux2, levels,
                                  def_types, ctor_tags, rec_rules,
                                  child1[base + root_a], child1[base + root_b], base, MN, pool_ptr, mvar_env, max_mvars);
            if (!f_eq) goto attempt_reduction;
            return unify_dev(node_types, child1, child2, child3, aux1, aux2, levels,
                             def_types, ctor_tags, rec_rules,
                             child2[base + root_a], child2[base + root_b], base, MN, pool_ptr, mvar_env, max_mvars);
        }

        if (type_a == N_PI || type_a == N_LAM) {
            bool dom_eq = unify_dev(node_types, child1, child2, child3, aux1, aux2, levels,
                                    def_types, ctor_tags, rec_rules,
                                    child2[base + root_a], child2[base + root_b], base, MN, pool_ptr, mvar_env, max_mvars);
            if (!dom_eq) goto attempt_reduction;
            return unify_dev(node_types, child1, child2, child3, aux1, aux2, levels,
                             def_types, ctor_tags, rec_rules,
                             child1[base + root_a], child1[base + root_b], base, MN, pool_ptr, mvar_env, max_mvars);
        }
    }

attempt_reduction:
    // If structural sharing and basic shape checking fails, they might be definitionally equal 
    // but not in Weak Head Normal Form. (e.g. `2 + 2` vs `4`)
    // Reduce both to WHNF and compare again.

    int dummy_steps = 0;
    int norm_a = whnf_single_dev(node_types, child1, child2, child3, aux1, aux2, levels, 
                                 def_types, ctor_tags, rec_rules, root_a, base, MN, pool_ptr, &dummy_steps);
    int norm_b = whnf_single_dev(node_types, child1, child2, child3, aux1, aux2, levels, 
                                 def_types, ctor_tags, rec_rules, root_b, base, MN, pool_ptr, &dummy_steps);

    // If reduction changed nothing, they are strictly not equal.
    if (norm_a == root_a && norm_b == root_b) return false;

    // Retry on the reduced forms. To avoid stack overflow, we just do one final shallow/recursive check.
    // A fully robust DefEq would maintain a bounded depth stack here.
    if (norm_a == norm_b) return true;

    int64_t norm_type_a = node_types[base + norm_a];
    int64_t norm_type_b = node_types[base + norm_b];

    if (norm_type_a != norm_type_b) return false;

    if (norm_type_a == N_APP) {
        bool f_eq = unify_dev(node_types, child1, child2, child3, aux1, aux2, levels,
                              def_types, ctor_tags, rec_rules,
                              child1[base + norm_a], child1[base + norm_b], base, MN, pool_ptr, mvar_env, max_mvars);
        if (!f_eq) return false;
        return unify_dev(node_types, child1, child2, child3, aux1, aux2, levels,
                         def_types, ctor_tags, rec_rules,
                         child2[base + norm_a], child2[base + norm_b], base, MN, pool_ptr, mvar_env, max_mvars);
    }

    return false;
}

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
    const int64_t* __restrict__ ctor_tags,    // [MAX_CONSTS] — constructor tags
    const int64_t* __restrict__ rec_rules,    // [MAX_CONSTS] — recursor rules
    int64_t* __restrict__ whnf_steps,
    int B, int MN
) {
    int bi = blockIdx.x * blockDim.x + threadIdx.x;
    if (bi >= B) return;

    int base = bi * MN;
    int64_t root = root_indices[bi];
    int steps = 0;
    int pool_ptr_val = MN - 1;
    int* pool_ptr = &pool_ptr_val;

    root = whnf_single_dev(node_types, child1, child2, child3, aux1, aux2, levels,
                           def_types, ctor_tags, rec_rules, root, base, MN, pool_ptr, &steps);

    root_indices[bi] = root;
    whnf_steps[bi] = steps;
}

// ============================================================
// DEVICE: Phase 9.1 Tactic Engine (Autonomous AST Synthesis)
// ============================================================

__device__ int tactic_apply_dev(
    int64_t* node_types, int64_t* child1, int64_t* child2,
    int64_t* child3, int64_t* aux1, int64_t* aux2, int64_t* levels,
    int func_ast_idx, int arg_ast_idx, int base, int MN, int* pool_ptr, int current_level
) {
    // Tactic 'apply': Takes a function and an argument, synthesizes an N_APP node on the GPU.
    // Equivalent to AST: App(func, arg)
    int64_t app_idx;
    ALLOC_NODE_HASH_CONS(N_APP, func_ast_idx, arg_ast_idx, -1, 0, 0, current_level, app_idx);
    return (int)app_idx;
}

__device__ int tactic_intro_dev(
    int64_t* node_types, int64_t* child1, int64_t* child2,
    int64_t* child3, int64_t* aux1, int64_t* aux2, int64_t* levels,
    int dom_ast_idx, int body_ast_idx, int binder_info, int base, int MN, int* pool_ptr, int current_level
) {
    // Tactic 'intro': Synthesizes a Lambda (N_LAM) node on the GPU.
    // Equivalent to AST: Lam(body, dom)
    int64_t lam_idx;
    ALLOC_NODE_HASH_CONS(N_LAM, body_ast_idx, dom_ast_idx, -1, binder_info, 0, current_level, lam_idx);
    return (int)lam_idx;
}


// ============================================================
// DEVICE: Phase 9.2 Warp-Cooperative Proof Search (MCTS Prototype)
// ============================================================

__device__ int synthesize_proof_dev(
    int64_t* node_types, int64_t* child1, int64_t* child2,
    int64_t* child3, int64_t* aux1, int64_t* aux2, int64_t* levels,
    const int64_t* def_types, const int64_t* ctor_tags, const int64_t* rec_rules,
    int goal_type_idx, int base, int MN, int* pool_ptr, int current_level, int seed
) {
    // A simplified proof synthesizer.
    // Given a goal type, it attempts to synthesize an AST (a proof term) that matches it.
    // It randomly selects tactics ('intro' or 'apply') based on a simple linear congruential generator.
    
    // Very basic LCG (Linear Congruential Generator) for randomness on GPU
    unsigned int state = seed + threadIdx.x;
    state = state * 1664525 + 1013904223;
    
    // Fallback if we cannot synthesize
    int current_proof_idx = -1;

    // Rule 1: If goal is Pi(A, B), we almost always want to use `intro` (Lam).
    if (node_types[base + goal_type_idx] == N_PI) {
        int64_t dom = child1[base + goal_type_idx];
        int64_t cod = child2[base + goal_type_idx];
        
        // Synthesize the body recursively, with the new context (B).
        // In a true implementation, we need to shift/manage De Bruijn indices correctly.
        int body_proof = synthesize_proof_dev(
            node_types, child1, child2, child3, aux1, aux2, levels,
            def_types, ctor_tags, rec_rules,
            cod, base, MN, pool_ptr, current_level, state
        );
        
        if (body_proof != -1) {
            current_proof_idx = tactic_intro_dev(
                node_types, child1, child2, child3, aux1, aux2, levels,
                dom, body_proof, 0, base, MN, pool_ptr, current_level
            );
        }
    } 
    else {
        // Goal is an atomic type (e.g. Nat, Prop, Eq a b).
        // We try to `apply` a known constant or a local variable.
        // For this prototype, we just "guess" a Nat.zero if goal is Nat.
        if (goal_type_idx == 2 /* Pre-allocated Nat */) {
            // Allocate a Nat.zero (Index 5 in environment, though we should map it dynamically)
            // Let's just create a generic constructor node.
            ALLOC_NODE_HASH_CONS(N_NAT_ZERO, -1, -1, -1, 5, 0, current_level, current_proof_idx);
        }
    }

    return current_proof_idx;
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

    // Phase 8.6: Cooperative Shared Memory Substitution Environment
    // We allocate a shared array for metavariables. All threads in this block
    // share this environment. If thread A finds ?m0 = Nat, thread B sees it instantly.
    __shared__ int64_t mvar_env[64]; // Max 64 metavariables per block for now
    if (threadIdx.x < 64) mvar_env[threadIdx.x] = 0;
    __syncthreads();

    // Track the end of the term for local allocations (e.g. Dependent Pattern Matching substitution)
    int pool_ptr_val = MN - 1;
    int* pool_ptr = &pool_ptr_val;

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
                res = (a1 == 0) ? 1 : ((a1 == 1) ? 2 /* T_TYPE1 missing, fallback to 1 for now if deeper */ : 1);
                // Phase 8.7: AST index returning
                // if a1 == 0 (Sort 0 / Prop), return Sort 1 (Type) -> which is pre-allocated at index 1!
                // if a1 == 1 (Sort 1 / Type), return Sort 2 (Type 1) -> we need to handle this.
                if (a1 == 0) res = 1; // Index of SORT 1
                else if (a1 == 1) {
                    int64_t type_idx;
                    ALLOC_NODE_HASH_CONS(N_SORT, -1, -1, -1, a1 + 1, 0, target_level, type_idx);
                    res = type_idx;
                }
                break;

            case N_VAR:
                res = (a2 != 0) ? a2 : a1;
                break;

            case N_CONST:
                if (a1 >= 0 && a1 < MAX_CONSTS) {
                    // Phase 8.7: Const types are now AST indices!
                    res = const_types[a1];
                }
                break;

            case N_NATLIT:
                res = 2; // Pre-allocated CONST Nat
                break;

            case N_STRLIT:
                res = 4; // Pre-allocated CONST String
                break;

            case N_LAM: {
                int64_t body_type = ct1;
                int64_t dom_type = ct2;

                // Phase 8.7: Domain is now an AST index!
                int64_t dom = dom_type;

                if (dom != T_ERROR && body_type != T_ERROR) {
                    // Return Pi(dom, body_type)
                    int64_t pi_idx;
                    ALLOC_NODE_HASH_CONS(N_PI, dom, body_type, -1, 0, 0, target_level, pi_idx);
                    res = pi_idx;
                }
                break;
            }

            case N_PI: {
                // Rule: Sort(imax(u, v))
                int64_t l1 = levels[base + ct1]; // Phase 8.7 levels of the domain AST
                int64_t l2 = levels[base + ct2];
                // Simplified for now: just return Sort(imax(0, 0)) -> Prop (0)
                int64_t sort_idx;
                ALLOC_NODE_HASH_CONS(N_SORT, -1, -1, -1, 0, 0, target_level, sort_idx);
                res = sort_idx;
                break;
            }

            case N_APP: {
                int64_t func_type = ct1;
                int64_t arg_type = ct2;

                if (func_type >= 0 && func_type < MN && node_types[base + func_type] == N_PI) {
                    int64_t dom = child1[base + func_type];
                    int64_t cod = child2[base + func_type];
                    
                    if (dom >= 0) {
                        bool type_match = unify_dev(
                            node_types, child1, child2, child3, aux1, aux2, levels,
                            nullptr, nullptr, nullptr,
                            (int)dom, (int)arg_type, base, MN, pool_ptr,
                            mvar_env, 64
                        );
                        
                        if (type_match) {
                            // Phase 8.7 Substitution: B[x:=a]
                            int64_t new_cod;
                            subst_alloc_dev(node_types, child1, child2, child3, aux1, aux2, levels,
                                            cod, 0, arg_type, base, MN, pool_ptr, &new_cod);
                            res = new_cod;
                        }
                    }
                }
                break;
            }

            case N_LET:
                res = ct3;
                break;

            case N_CTOR: {
                // Phase 8.7: Constructor result type should be computed from the inductive type definition.
                // For now, it's fetched from Python pre-computation via a2 (if we passed the AST index).
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
// DEVICE: Phase 10.1 The Algebraic Oracle (Eq Discovery)
// ============================================================

__device__ int deep_copy_dev(
    int64_t* node_types, int64_t* child1, int64_t* child2,
    int64_t* child3, int64_t* aux1, int64_t* aux2, int64_t* levels,
    int src_root_idx, int base, int MN, int* pool_ptr, int current_level
) {
    if (src_root_idx < 0 || src_root_idx >= MN) return src_root_idx;
    
    int64_t ntype = node_types[base + src_root_idx];
    
    // Primitives/Leaves are immutable and shared
    if (ntype == N_SORT || ntype == N_CONST || ntype == N_VAR || ntype == N_STRLIT || 
        ntype == N_NATLIT || ntype == N_NAT_ZERO || ntype == N_BOOL_TRUE || ntype == N_BOOL_FALSE) {
        return src_root_idx;
    }
    
    // Recursive copy for compound nodes
    int64_t c1 = child1[base + src_root_idx];
    int64_t c2 = child2[base + src_root_idx];
    int64_t c3 = child3[base + src_root_idx];
    
    int new_c1 = c1;
    int new_c2 = c2;
    int new_c3 = c3;
    
    // Shallow recursion (inlined by NVCC mostly, but beware of depth limits)
    if (c1 >= 0 && c1 < MN) new_c1 = deep_copy_dev(node_types, child1, child2, child3, aux1, aux2, levels, c1, base, MN, pool_ptr, current_level);
    if (c2 >= 0 && c2 < MN) new_c2 = deep_copy_dev(node_types, child1, child2, child3, aux1, aux2, levels, c2, base, MN, pool_ptr, current_level);
    if (c3 >= 0 && c3 < MN) new_c3 = deep_copy_dev(node_types, child1, child2, child3, aux1, aux2, levels, c3, base, MN, pool_ptr, current_level);
    
    int64_t new_node_idx;
    ALLOC_NODE_HASH_CONS(ntype, new_c1, new_c2, new_c3, aux1[base + src_root_idx], aux2[base + src_root_idx], current_level, new_node_idx);
    
    return new_node_idx;
}

__global__ void engine_mutate_kernel(
    int64_t* __restrict__ node_types, int64_t* __restrict__ child1,
    int64_t* __restrict__ child2, int64_t* __restrict__ child3,
    int64_t* __restrict__ aux1, int64_t* __restrict__ aux2,
    int64_t* __restrict__ levels, int64_t* __restrict__ root_indices,
    const int64_t* __restrict__ archive_roots, // Pointers to elite trees
    int archive_size, int B, int MN, int seed
) {
    int bi = blockIdx.x * blockDim.x + threadIdx.x;
    if (bi >= B) return;

    int base = bi * MN;
    
    // Very basic LCG
    unsigned int state = seed + bi * 1999;
    state = state * 1664525 + 1013904223;

    // Reset tree to primitives
    for (int i = 5; i < MN; i++) {
        node_types[base + i] = N_NONE;
    }

    int pool_ptr_val = MN - 1;
    int* pool_ptr = &pool_ptr_val;
    int current_level = 0;
    
    // Select 1 parent tree from archive
    int p1_idx = -1;
    if (archive_size > 0) {
        state = state * 1664525 + 1013904223;
        int archive_id = state % archive_size;
        p1_idx = archive_roots[archive_id]; 
    }
    
    state = state * 1664525 + 1013904223;
    bool random_gen = (archive_size == 0 || (state % 100 < 50));

    int new_root = -1;

    // Phase 10.1 The Algebraic Oracle: Eq(a, b) focus
    // Eq.refl : (a : A) -> Eq A a a
    // We synthesize App(Eq.refl, a) to prove Eq A a a.
    
    // We need to know the IDs of Eq.refl and Nat constants. 
    // In our environment: Nat=2 (preallocated), Eq.refl is roughly index 34.
    // For this kernel to be truly generic we'd pass these IDs from Python, but we'll use assumed IDs for the prototype.
    int EQ_REFL_CID = 34; 
    int NAT_ZERO_CID = 5;
    int NAT_SUCC_CID = 6;
    int NAT_ADD_CID = 7;
    int NAT_MUL_CID = 8;
    int NAT_SUB_CID = 9;

    if (random_gen || p1_idx == -1) {
        // Synthesize a random algebraic term (a)
        state = state * 1664525 + 1013904223;
        int action = state % 3;
        
        int64_t term_a_idx = -1;
        
        if (action == 0) {
            // Nat.zero
            ALLOC_NODE_HASH_CONS(N_CONST, -1, -1, -1, NAT_ZERO_CID, 0, current_level, term_a_idx);
        } 
        else if (action == 1) {
            // Nat.succ Nat.zero
            int64_t z_node;
            ALLOC_NODE_HASH_CONS(N_CONST, -1, -1, -1, NAT_ZERO_CID, 0, current_level, z_node);
            int64_t s_node;
            ALLOC_NODE_HASH_CONS(N_CONST, -1, -1, -1, NAT_SUCC_CID, 0, current_level, s_node);
            ALLOC_NODE_HASH_CONS(N_APP, s_node, z_node, -1, 0, 0, current_level, term_a_idx);
        }
        else {
            // Nat.add Nat.zero Nat.zero
            int64_t z_node;
            ALLOC_NODE_HASH_CONS(N_CONST, -1, -1, -1, NAT_ZERO_CID, 0, current_level, z_node);
            int64_t a_node;
            ALLOC_NODE_HASH_CONS(N_CONST, -1, -1, -1, NAT_ADD_CID, 0, current_level, a_node);
            
            int64_t app1;
            ALLOC_NODE_HASH_CONS(N_APP, a_node, z_node, -1, 0, 0, current_level, app1);
            ALLOC_NODE_HASH_CONS(N_APP, app1, z_node, -1, 0, 0, current_level, term_a_idx);
        }
        
        // Now wrap the term `a` in `Eq.refl Nat a` to prove `Eq Nat a a`
        if (term_a_idx != -1) {
            int64_t refl_node;
            ALLOC_NODE_HASH_CONS(N_CONST, -1, -1, -1, EQ_REFL_CID, 0, current_level, refl_node);
            
            // App(Eq.refl, term_a_idx) -> we ignore implicit `Nat` argument for simplicity in this AST builder 
            // depending on how Lean parses it. If `Nat` is explicit, we need App(App(Eq.refl, Nat), a)
            int64_t nat_node = 2; // Preallocated CONST Nat
            int64_t app_refl_nat;
            ALLOC_NODE_HASH_CONS(N_APP, refl_node, nat_node, -1, 0, 0, current_level, app_refl_nat);
            ALLOC_NODE_HASH_CONS(N_APP, app_refl_nat, term_a_idx, -1, 0, 0, current_level, new_root);
        }
    } 
    else {
        // Crossover/Mutate from Algebraic Parent
        int64_t p1_local_root = deep_copy_dev(node_types, child1, child2, child3, aux1, aux2, levels, p1_idx, base, MN, pool_ptr, current_level);
        
        // Let's wrap the parent theorem `p` in `Eq.symm` (p : Eq A a b -> Eq A b a)
        int EQ_SYMM_CID = 35;
        int64_t symm_node;
        ALLOC_NODE_HASH_CONS(N_CONST, -1, -1, -1, EQ_SYMM_CID, 0, current_level, symm_node);
        
        // App(Eq.symm, p)
        ALLOC_NODE_HASH_CONS(N_APP, symm_node, p1_local_root, -1, 0, 0, current_level, new_root);
    }

    if (new_root == -1) {
        new_root = 2; 
    }

    root_indices[bi] = new_root;
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
    int blocks = (B + threads - 1) / threads;

    // Phase 1: WHNF reduction with proper substitution
    engine_whnf_kernel<<<blocks, threads>>>(
        node_types.data_ptr<int64_t>(), child1.data_ptr<int64_t>(),
        child2.data_ptr<int64_t>(), child3.data_ptr<int64_t>(),
        aux1.data_ptr<int64_t>(), aux2.data_ptr<int64_t>(),
        levels.data_ptr<int64_t>(), root_indices.data_ptr<int64_t>(),
        def_types.data_ptr<int64_t>(), ctor_tags.data_ptr<int64_t>(), 
        rec_rules.data_ptr<int64_t>(), whnf_steps.data_ptr<int64_t>(),
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


// ============================================================
// KERNEL: Cooperative MCTS Proof Search (Phase 9.3)
// ============================================================

__global__ void engine_search_kernel(
    int64_t* __restrict__ node_types, int64_t* __restrict__ child1,
    int64_t* __restrict__ child2, int64_t* __restrict__ child3,
    int64_t* __restrict__ aux1, int64_t* __restrict__ aux2,
    int64_t* __restrict__ levels, int64_t* __restrict__ root_indices,
    const int64_t* __restrict__ def_types, const int64_t* __restrict__ ctor_tags,
    const int64_t* __restrict__ rec_rules,
    int target_level, int B, int MN
) {
    int bi = blockIdx.x; // One Proof Term per Block (Cooperative)
    if (bi >= B) return;

    int base = bi * MN;
    int goal_root = root_indices[bi];
    
    // Shared Memory Proof Board (Phase 9.3)
    __shared__ int goal_type_idx[256];
    __shared__ int proof_term_idx[256];
    __shared__ int parent_goal[256];
    __shared__ int queue_tail;
    
    if (threadIdx.x == 0) {
        queue_tail = 1;
        goal_type_idx[0] = goal_root; // Root Goal
        proof_term_idx[0] = -1;       // Not proved yet
        parent_goal[0] = -1;          // No parent
    }
    __syncthreads();

    // Track the end of the term for local allocations
    int pool_ptr_val = MN - 1;
    int* pool_ptr = &pool_ptr_val;

    // A simple Monte Carlo loop (to be expanded with actual Backprop)
    int max_mcts_steps = 100;
    int step = 0;
    
    unsigned int state = bi * 1999 + threadIdx.x; // Seed
    
    while (proof_term_idx[0] == -1 && step < max_mcts_steps) {
        state = state * 1664525 + 1013904223; // LCG
        
        // Pick a random unproved goal from the queue
        int q_size = queue_tail;
        if (q_size > 0) {
            int random_goal_idx = state % q_size;
            
            if (proof_term_idx[random_goal_idx] == -1) {
                int current_goal_ast = goal_type_idx[random_goal_idx];
                
                // Attempt to synthesize
                int found_proof = synthesize_proof_dev(
                    node_types, child1, child2, child3, aux1, aux2, levels,
                    def_types, ctor_tags, rec_rules,
                    current_goal_ast, base, MN, pool_ptr, target_level, state
                );
                
                if (found_proof != -1) {
                    // Try to submit proof (Atomic to prevent overwriting)
                    atomicCAS(&proof_term_idx[random_goal_idx], -1, found_proof);
                }
            }
        }
        
        step++;
        __syncthreads();
    }
    
    // If root is proved, thread 0 updates the final root_indices
    if (threadIdx.x == 0 && proof_term_idx[0] != -1) {
        root_indices[bi] = proof_term_idx[0];
    }
}

// ============================================================
// HOST: MCTS Search Pipeline
// ============================================================

std::vector<torch::Tensor> cic_engine_search(
    torch::Tensor node_types, torch::Tensor child1, torch::Tensor child2,
    torch::Tensor child3, torch::Tensor aux1, torch::Tensor aux2,
    torch::Tensor levels, torch::Tensor root_indices,
    torch::Tensor pi_lookup, torch::Tensor const_types,
    torch::Tensor def_types, torch::Tensor ctor_tags, torch::Tensor rec_rules,
    int max_level
) {
    int B = node_types.size(0);
    int MN = node_types.size(1);

    // One Block per Theorem. 256 Threads cooperating.
    int threads = 256;
    int blocks = B; 

    engine_search_kernel<<<blocks, threads>>>(
        node_types.data_ptr<int64_t>(), child1.data_ptr<int64_t>(),
        child2.data_ptr<int64_t>(), child3.data_ptr<int64_t>(),
        aux1.data_ptr<int64_t>(), aux2.data_ptr<int64_t>(),
        levels.data_ptr<int64_t>(), root_indices.data_ptr<int64_t>(),
        def_types.data_ptr<int64_t>(), ctor_tags.data_ptr<int64_t>(), 
        rec_rules.data_ptr<int64_t>(),
        max_level, B, MN
    );
    
    // After search, run the standard type checker to verify the synthesized proofs
    return cic_engine_pipeline(
        node_types, child1, child2, child3, aux1, aux2,
        levels, root_indices, pi_lookup, const_types,
        def_types, ctor_tags, rec_rules, max_level
    );
}


// ============================================================
// HOST: MAP-Elites Evolution Pipeline (Phase 10)
// ============================================================

std::vector<torch::Tensor> cic_engine_mutate(
    torch::Tensor node_types, torch::Tensor child1, torch::Tensor child2,
    torch::Tensor child3, torch::Tensor aux1, torch::Tensor aux2,
    torch::Tensor levels, torch::Tensor root_indices,
    torch::Tensor pi_lookup, torch::Tensor const_types,
    torch::Tensor def_types, torch::Tensor ctor_tags, torch::Tensor rec_rules,
    torch::Tensor archive_roots, int archive_size, int max_level, int seed
) {
    int B = node_types.size(0);
    int MN = node_types.size(1);

    int threads = 256;
    int blocks = (B + threads - 1) / threads;

    // Mutate trees
    engine_mutate_kernel<<<blocks, threads>>>(
        node_types.data_ptr<int64_t>(), child1.data_ptr<int64_t>(),
        child2.data_ptr<int64_t>(), child3.data_ptr<int64_t>(),
        aux1.data_ptr<int64_t>(), aux2.data_ptr<int64_t>(),
        levels.data_ptr<int64_t>(), root_indices.data_ptr<int64_t>(),
        archive_roots.data_ptr<int64_t>(), archive_size, B, MN, seed
    );

    // After mutation, run the standard type checker to verify the new ASTs
    return cic_engine_pipeline(
        node_types, child1, child2, child3, aux1, aux2,
        levels, root_indices, pi_lookup, const_types,
        def_types, ctor_tags, rec_rules, max_level
    );
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
    m.def("cic_engine_search", &cic_engine_search,
          "Phase 9.3: Autonomous MCTS Search Engine");
    m.def("cic_engine_mutate", &cic_engine_mutate,
          "Phase 10: Evolutionary AST Mutator");
    m.def("cic_gpu_type_check", &cic_gpu_type_check,
          "Backward-compatible type check interface");
}
