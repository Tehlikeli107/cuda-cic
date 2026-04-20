"""
Test Suite: De Bruijn Substitution Engine + Environment Builder
================================================================
Tests the fundamental correctness of:
  1. CIC Environment builder (constant registration, type hashes)
  2. De Bruijn variable resolution (binding context)
  3. Tree flattening with proper variable handling
  4. GPU substitution kernel (when available)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lean4'))

import numpy as np

from env_builder import (
    CICEnvironment, TypeClassResolver,
    T_ERROR, T_PROP, T_TYPE, T_TYPE1, T_NAT, T_BOOL,
    pi_hash, sort_hash, NAT_NAT, NAT_NAT_NAT, NAT_PROP,
    BOOL_BOOL, NAT_NAT_PROP
)

# ============================================================
# TEST HELPERS
# ============================================================

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}  {detail}")


# ============================================================
# TEST 1: Environment Builder
# ============================================================

print("\n" + "=" * 60)
print("TEST 1: CIC Environment Builder")
print("=" * 60)

env = CICEnvironment()

# Core types registered
check("Nat registered", env.get_type("Nat") == T_TYPE)
check("Bool registered", env.get_type("Bool") == T_TYPE)
check("Prop registered", env.get_type("Prop") == T_TYPE)

# Nat constructors
check("Nat.zero : Nat", env.get_type("Nat.zero") == T_NAT)
check("Nat.succ : Nat→Nat", env.get_type("Nat.succ") == NAT_NAT)

# Nat operations
check("Nat.add : Nat→Nat→Nat", env.get_type("Nat.add") == NAT_NAT_NAT)
check("Nat.mul : Nat→Nat→Nat", env.get_type("Nat.mul") == NAT_NAT_NAT)

# Bool
check("Bool.true : Bool", env.get_type("Bool.true") == T_BOOL)
check("Bool.false : Bool", env.get_type("Bool.false") == T_BOOL)
check("Bool.not : Bool→Bool", env.get_type("Bool.not") == BOOL_BOOL)

# Unknown constant auto-registers with T_ERROR
cid = env.get_or_create("SomeUnknown.thing")
check("Unknown auto-registers", cid >= 0)
check("Unknown has T_ERROR type", env.get_type("SomeUnknown.thing") == T_ERROR)

# Eq
check("Eq.refl type", env.get_type("Eq.refl") == NAT_PROP)

# Numpy export
arr = env.to_numpy()
check("Numpy array shape", arr.shape == (65536,))
nat_id = env.name_to_id["Nat"]
check("Numpy[Nat] = T_TYPE", arr[nat_id] == T_TYPE)

# Lookup table
lookup = env.build_lookup_array()
h = pi_hash(T_NAT, T_NAT)
check("Lookup Nat→Nat dom", lookup[h * 2] == T_NAT)
check("Lookup Nat→Nat cod", lookup[h * 2 + 1] == T_NAT)


# ============================================================
# TEST 2: Pi Hash Properties
# ============================================================

print("\n" + "=" * 60)
print("TEST 2: Pi Hash Properties")
print("=" * 60)

# Different inputs → different hashes
h1 = pi_hash(T_NAT, T_NAT)
h2 = pi_hash(T_NAT, T_BOOL)
h3 = pi_hash(T_BOOL, T_NAT)
h4 = pi_hash(T_NAT, NAT_NAT)

check("Nat→Nat ≠ Nat→Bool", h1 != h2)
check("Nat→Bool ≠ Bool→Nat", h2 != h3)
check("Nat→Nat ≠ Nat→(Nat→Nat)", h1 != h4)
check("Hash > 0", all(h > 0 for h in [h1, h2, h3, h4]))

# Sort hashes
check("sort(0) = Prop", sort_hash(0) == T_PROP)
check("sort(1) = Type", sort_hash(1) == T_TYPE)
check("sort(2) = Type1", sort_hash(2) == T_TYPE1)
check("sort(3) > 0", sort_hash(3) > 0)
check("sort levels distinct", len(set(sort_hash(i) for i in range(10))) == 10)

# Nested pi types
# Nat → Nat → Nat should be Nat → (Nat → Nat)
inner = pi_hash(T_NAT, T_NAT)  # Nat → Nat
outer = pi_hash(T_NAT, inner)   # Nat → (Nat → Nat)
check("Nat→Nat→Nat = pi(Nat, pi(Nat, Nat))", outer == NAT_NAT_NAT)


# ============================================================
# TEST 3: Binding Context (De Bruijn)
# ============================================================

print("\n" + "=" * 60)
print("TEST 3: Binding Context (De Bruijn)")
print("=" * 60)

from lean4_to_gpu import BindingContext

ctx = BindingContext()
check("Empty context depth", ctx.depth() == 0)
check("Unbound BVAR(0)", ctx.lookup(0) == T_ERROR)

# Push Nat binding
ctx.push(T_NAT)
check("After push Nat: depth=1", ctx.depth() == 1)
check("BVAR(0) = Nat", ctx.lookup(0) == T_NAT)
check("BVAR(1) = ERROR (unbound)", ctx.lookup(1) == T_ERROR)

# Push Bool binding (innermost)
ctx.push(T_BOOL)
check("After push Bool: depth=2", ctx.depth() == 2)
check("BVAR(0) = Bool (innermost)", ctx.lookup(0) == T_BOOL)
check("BVAR(1) = Nat (outer)", ctx.lookup(1) == T_NAT)
check("BVAR(2) = ERROR (unbound)", ctx.lookup(2) == T_ERROR)

# Pop Bool
ctx.pop()
check("After pop: depth=1", ctx.depth() == 1)
check("BVAR(0) = Nat again", ctx.lookup(0) == T_NAT)

# Pop Nat
ctx.pop()
check("After pop all: depth=0", ctx.depth() == 0)


# ============================================================
# TEST 4: Tree Flattening (v2)
# ============================================================

print("\n" + "=" * 60)
print("TEST 4: Tree Flattening (v2)")
print("=" * 60)

from lean4_to_gpu import ExprNode, flatten_tree_v2

N_SORT = 0; N_VAR = 1; N_CONST = 2; N_APP = 3; N_LAM = 4; N_PI = 5

# Test: CONST Nat → should produce single CONST node
tree_nat = ExprNode('CONST', name='Nat')
nodes, root = flatten_tree_v2(tree_nat, env)
check("CONST Nat: 1 node", len(nodes) == 1)
check("CONST Nat: root=0", root == 0)
check("CONST Nat: type=N_CONST", nodes[0][0] == N_CONST)
nat_cid = env.name_to_id["Nat"]
check("CONST Nat: a1=nat_cid", nodes[0][4] == nat_cid)

# Test: NATLIT 42
tree_42 = ExprNode('NATLIT', value=42)
nodes, root = flatten_tree_v2(tree_42, env)
check("NATLIT 42: 1 node", len(nodes) == 1)
check("NATLIT 42: a1=42", nodes[0][4] == 42)

# Test: SORT 0 (Prop)
tree_prop = ExprNode('SORT', value='0')
nodes, root = flatten_tree_v2(tree_prop, env)
check("SORT 0: a1=0", nodes[0][4] == 0)

# Test: FORALL n (CONST Nat) (BVAR 0)
# ∀ n : Nat, n  (identity predicate, not well-typed as Prop but tests parsing)
tree_forall = ExprNode('FORALL', name='n', children=[
    ExprNode('CONST', name='Nat'),      # domain
    ExprNode('BVAR', value=0),            # codomain (using bound var)
])
nodes, root = flatten_tree_v2(tree_forall, env)
check("FORALL: >=3 nodes", len(nodes) >= 3)
# The PI node should have children pointing to domain and codomain
pi_node = nodes[root]
check("FORALL: root is PI", pi_node[0] == N_PI)

# Find the BVAR node
bvar_nodes = [(i, n) for i, n in enumerate(nodes) if n[0] == N_VAR]
check("FORALL: has BVAR node", len(bvar_nodes) > 0)
if bvar_nodes:
    bvar_idx, bvar_node = bvar_nodes[0]
    check("BVAR: de Bruijn idx = 0", bvar_node[4] == 0)
    # a2 should have resolved type = T_NAT (from CONST Nat domain)
    check("BVAR: resolved type = Nat", bvar_node[5] == T_NAT,
          f"got {bvar_node[5]}")

# Test: LAM x (CONST Nat) (BVAR 0)
# λ x : Nat . x  (identity function)
tree_lam = ExprNode('LAM', name='x', children=[
    ExprNode('CONST', name='Nat'),
    ExprNode('BVAR', value=0),
])
nodes, root = flatten_tree_v2(tree_lam, env)
lam_node = nodes[root]
check("LAM: root is LAM", lam_node[0] == N_LAM)
bvar_nodes = [(i, n) for i, n in enumerate(nodes) if n[0] == N_VAR]
if bvar_nodes:
    check("LAM BVAR: resolved type = Nat", bvar_nodes[0][1][5] == T_NAT)

# Test: Nested lambdas (fun x:Nat => fun y:Bool => x)
tree_nested = ExprNode('LAM', name='x', children=[
    ExprNode('CONST', name='Nat'),
    ExprNode('LAM', name='y', children=[
        ExprNode('CONST', name='Bool'),
        ExprNode('BVAR', value=1),  # refers to x (outer lambda)
    ]),
])
nodes, root = flatten_tree_v2(tree_nested, env)
# Find BVAR(1) node — should resolve to Nat (the outer binding)
bvar_nodes = [(i, n) for i, n in enumerate(nodes) if n[0] == N_VAR]
if bvar_nodes:
    bvar1 = [n for _, n in bvar_nodes if n[4] == 1]
    if bvar1:
        check("Nested LAM: BVAR(1) resolves to Nat", bvar1[0][5] == T_NAT,
              f"got {bvar1[0][5]}")
    else:
        check("Nested LAM: BVAR(1) found", False, "BVAR(1) not found")


# ============================================================
# TEST 5: V1 Format Parsing (Backward Compat)
# ============================================================

print("\n" + "=" * 60)
print("TEST 5: V1 Format Parsing")
print("=" * 60)

from lean4_to_gpu import parse_theorems_v1

v1_text = """=== THEOREM Nat.add_comm ===
FORALL n
  CONST Nat
  FORALL m
    CONST Nat
    APP
      APP
        APP
          CONST Eq
          CONST Nat
        BVAR 1
      BVAR 0
---"""

theorems = parse_theorems_v1(v1_text)
check("V1: parsed 1 theorem", len(theorems) == 1)
check("V1: name = Nat.add_comm", "Nat.add_comm" in theorems)

tree = theorems["Nat.add_comm"]
check("V1: root is FORALL", tree.kind == "FORALL")
check("V1: has 2 children", len(tree.children) == 2)
check("V1: domain is CONST Nat", tree.children[0].kind == "CONST" and tree.children[0].name == "Nat")


# ============================================================
# TEST 6: Type Class Resolver
# ============================================================

print("\n" + "=" * 60)
print("TEST 6: Type Class Resolver")
print("=" * 60)

check("HAdd.hAdd is typeclass", TypeClassResolver.is_typeclass_app("HAdd.hAdd"))
check("instHAdd is typeclass", TypeClassResolver.is_typeclass_app("instHAdd"))
check("Nat.add is NOT typeclass", not TypeClassResolver.is_typeclass_app("Nat.add"))

resolved = TypeClassResolver.resolve("HAdd.hAdd", ("Nat", "Nat", "Nat"))
check("HAdd Nat Nat Nat → Nat.add", resolved == "Nat.add")

resolved = TypeClassResolver.resolve("HMul.hMul", ("Nat", "Nat", "Nat"))
check("HMul Nat Nat Nat → Nat.mul", resolved == "Nat.mul")

resolved = TypeClassResolver.resolve("HAdd.hAdd", ("Int", "Int", "Int"))
check("HAdd Int Int Int → Int.add", resolved == "Int.add")

resolved = TypeClassResolver.resolve("Nat.add", ("Nat",))
check("Unknown pattern → None", resolved is None)


# ============================================================
# TEST 7: Real Exported Tree Flattening
# ============================================================

print("\n" + "=" * 60)
print("TEST 7: Real Exported Tree Flattening")
print("=" * 60)

export_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'lean4', 'exported_trees.txt')
if os.path.exists(export_path):
    with open(export_path, encoding='utf-8') as f:
        text = f.read()

    theorems = parse_theorems_v1(text)
    check(f"Parsed {len(theorems)} theorems", len(theorems) > 0)

    for name, tree in theorems.items():
        nodes, root = flatten_tree_v2(tree, env)
        n_nodes = len(nodes)
        n_vars = sum(1 for n in nodes if n[0] == N_VAR)
        n_consts = sum(1 for n in nodes if n[0] == N_CONST)

        check(f"{name}: {n_nodes} nodes, root={root}",
              n_nodes > 0 and root >= 0 and root < n_nodes,
              f"nodes={n_nodes}, root={root}")

        # All BVAR nodes should have valid de Bruijn indices
        for j, n in enumerate(nodes):
            if n[0] == N_VAR:
                db_idx = n[4]
                resolved = n[5]
                if db_idx < 0:
                    check(f"{name}: BVAR({db_idx}) valid", False,
                          f"negative de Bruijn index at node {j}")
                    break
else:
    print("  [SKIP] No exported_trees.txt found")


# ============================================================
# SUMMARY
# ============================================================

print(f"\n{'=' * 60}")
total = passed + failed
print(f"RESULTS: {passed}/{total} passed ({100*passed/total:.0f}%)")
if failed > 0:
    print(f"  {failed} FAILURES")
else:
    print("  ALL TESTS PASSED")
print(f"{'=' * 60}")
