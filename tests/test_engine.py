"""
Integration Test: Unified CIC Engine
======================================
Tests the cic_engine.cu unified kernel that combines
WHNF + substitution + type checking in one pipeline.

Requires: NVIDIA GPU with CUDA, PyTorch with CUDA support.
"""
import sys, io, os, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
os.environ['TORCH_EXTENSIONS_DIR'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_ext_cache')

import torch
import numpy as np

DEVICE = torch.device('cuda')
WORKDIR = os.path.dirname(os.path.abspath(__file__))

print(f"Device: {torch.cuda.get_device_name(0)}")

# ============================================================
# COMPILE UNIFIED ENGINE
# ============================================================
print("Compiling unified CIC engine...")
from torch.utils.cpp_extension import load
engine = load(
    name="cic_engine",
    sources=[os.path.join(os.path.dirname(WORKDIR), "kernels", "cic_engine.cu")],
    verbose=False
)
print("OK\n")

# ============================================================
# CONSTANTS (from env_builder)
# ============================================================
sys.path.insert(0, os.path.join(os.path.dirname(WORKDIR), 'lean4'))
from env_builder import (
    CICEnvironment, get_default_env,
    T_ERROR, T_PROP, T_TYPE, T_TYPE1, T_NAT, T_BOOL,
    pi_hash, MAX_CONSTS, TABLE_SIZE,
    NAT_NAT, NAT_NAT_NAT, BOOL_BOOL, NAT_PROP
)

env = get_default_env()
const_types_gpu = torch.from_numpy(env.to_numpy()).to(DEVICE)
lookup_gpu = torch.from_numpy(env.build_lookup_array()).to(DEVICE)
def_types_gpu = torch.zeros(MAX_CONSTS, dtype=torch.long, device=DEVICE)

# Node types
N_SORT=0; N_VAR=1; N_CONST=2; N_APP=3; N_LAM=4; N_PI=5; N_LET=6
N_NAT_ZERO=10; N_NAT_SUCC=11; N_NATLIT=9; N_BOOL_TRUE=12; N_BOOL_FALSE=13; N_NONE=-1

# Constant IDs
C_NAT_SUCC = env.name_to_id["Nat.succ"]
C_NAT_ADD = env.name_to_id["Nat.add"]
C_NAT_MUL = env.name_to_id["Nat.mul"]

def node(ntype, c1=-1, c2=-1, c3=-1, a1=0, a2=0, level=0):
    return (ntype, c1, c2, c3, a1, a2, level)


# ============================================================
# TEST CASES
# ============================================================

def gen_test_cases():
    proofs = []
    names = []
    expected = []

    # --- Sort hierarchy ---
    proofs.append(([node(N_SORT, a1=0)], 0))
    names.append("Sort(0) : Type"); expected.append(T_TYPE)

    proofs.append(([node(N_SORT, a1=1)], 0))
    names.append("Sort(1) : Type1"); expected.append(T_TYPE1)

    # --- Nat constructors ---
    proofs.append(([node(N_NAT_ZERO, level=0)], 0))
    names.append("0 : Nat"); expected.append(T_NAT)

    proofs.append(([node(N_NAT_ZERO, level=0),
                    node(N_NAT_SUCC, c1=0, level=1)], 1))
    names.append("S(0) : Nat"); expected.append(T_NAT)

    proofs.append(([node(N_NAT_ZERO, level=0),
                    node(N_NAT_SUCC, c1=0, level=1),
                    node(N_NAT_SUCC, c1=1, level=2)], 2))
    names.append("S(S(0)) : Nat"); expected.append(T_NAT)

    proofs.append(([node(N_NATLIT, a1=42)], 0))
    names.append("42 : Nat"); expected.append(T_NAT)

    # --- Bool ---
    proofs.append(([node(N_BOOL_TRUE, level=0)], 0))
    names.append("true : Bool"); expected.append(T_BOOL)

    proofs.append(([node(N_BOOL_FALSE, level=0)], 0))
    names.append("false : Bool"); expected.append(T_BOOL)

    # --- Constants ---
    proofs.append(([node(N_CONST, a1=C_NAT_SUCC)], 0))
    names.append("Nat.succ : Nat->Nat"); expected.append(NAT_NAT)

    proofs.append(([node(N_CONST, a1=C_NAT_ADD)], 0))
    names.append("Nat.add : Nat->Nat->Nat"); expected.append(NAT_NAT_NAT)

    # --- Lambda: id_Nat = fun x:Nat => x ---
    # N_VAR with a2 = T_NAT (resolved type)
    proofs.append(([node(N_VAR, a1=0, a2=T_NAT, level=0),
                    node(N_LAM, c1=0, a1=T_NAT, level=1)], 1))
    names.append("id_Nat : Nat->Nat"); expected.append(NAT_NAT)

    # --- Lambda: const = fun x:Nat => fun y:Nat => x ---
    proofs.append(([node(N_VAR, a1=1, a2=T_NAT, level=0),
                    node(N_LAM, c1=0, a1=T_NAT, level=1),
                    node(N_LAM, c1=1, a1=T_NAT, level=2)], 2))
    names.append("const : Nat->Nat->Nat"); expected.append(NAT_NAT_NAT)

    # --- Application: id_Nat 0 ---
    proofs.append(([node(N_VAR, a1=0, a2=T_NAT, level=0),
                    node(N_LAM, c1=0, a1=T_NAT, level=1),
                    node(N_NAT_ZERO, level=0),
                    node(N_APP, c1=1, c2=2, level=2)], 3))
    names.append("id_Nat 0 : Nat"); expected.append(T_NAT)

    # --- Application: Nat.succ 0 ---
    proofs.append(([node(N_CONST, a1=C_NAT_SUCC, level=0),
                    node(N_NAT_ZERO, level=0),
                    node(N_APP, c1=0, c2=1, level=1)], 2))
    names.append("Nat.succ 0 : Nat"); expected.append(T_NAT)

    # --- Partial application: Nat.add 0 ---
    proofs.append(([node(N_CONST, a1=C_NAT_ADD, level=0),
                    node(N_NAT_ZERO, level=0),
                    node(N_APP, c1=0, c2=1, level=1)], 2))
    names.append("Nat.add 0 : Nat->Nat"); expected.append(NAT_NAT)

    # --- Full application: Nat.add 0 0 ---
    proofs.append(([node(N_CONST, a1=C_NAT_ADD, level=0),
                    node(N_NAT_ZERO, level=0),
                    node(N_APP, c1=0, c2=1, level=1),
                    node(N_NAT_ZERO, level=0),
                    node(N_APP, c1=2, c2=3, level=2)], 4))
    names.append("Nat.add 0 0 : Nat"); expected.append(T_NAT)

    # --- Let binding: let x = 0 in S(x) ---
    proofs.append(([node(N_NAT_ZERO, level=0),
                    node(N_NAT_SUCC, c1=0, level=1),
                    node(N_LET, c1=0, c3=1, a1=T_NAT, level=2)], 2))
    names.append("let x=0 in S(x) : Nat"); expected.append(T_NAT)

    # --- fun x:Nat => S(S(x)) ---
    proofs.append(([node(N_VAR, a1=0, a2=T_NAT, level=0),
                    node(N_NAT_SUCC, c1=0, level=1),
                    node(N_NAT_SUCC, c1=1, level=2),
                    node(N_LAM, c1=2, a1=T_NAT, level=3)], 3))
    names.append("fun x => S(S(x)) : Nat->Nat"); expected.append(NAT_NAT)

    # --- INVALID: S(true) ---
    proofs.append(([node(N_BOOL_TRUE, level=0),
                    node(N_NAT_SUCC, c1=0, level=1)], 1))
    names.append("INVALID: S(true)"); expected.append(T_ERROR)

    # --- INVALID: App(0, 0) ---
    proofs.append(([node(N_NAT_ZERO, level=0),
                    node(N_NAT_ZERO, level=0),
                    node(N_APP, c1=0, c2=1, level=1)], 2))
    names.append("INVALID: App(0,0)"); expected.append(T_ERROR)

    return proofs, names, expected


# ============================================================
# BUILD BATCH & RUN
# ============================================================

def build_batch(proofs, max_nodes=32):
    B = len(proofs)
    MN = max_nodes
    nt = np.full((B, MN), N_NONE, dtype=np.int64)
    c1 = np.zeros((B, MN), dtype=np.int64)
    c2 = np.zeros((B, MN), dtype=np.int64)
    c3 = np.zeros((B, MN), dtype=np.int64)
    a1 = np.zeros((B, MN), dtype=np.int64)
    a2 = np.zeros((B, MN), dtype=np.int64)
    lv = np.full((B, MN), -1, dtype=np.int64)
    roots = np.zeros(B, dtype=np.int64)
    max_level = 0
    for i, (nodes, root) in enumerate(proofs):
        roots[i] = root
        for j, (ntype, cc1, cc2, cc3, aa1, aa2, level) in enumerate(nodes):
            if j >= MN: break
            nt[i,j] = ntype; c1[i,j] = max(cc1,0); c2[i,j] = max(cc2,0)
            c3[i,j] = max(cc3,0); a1[i,j] = aa1; a2[i,j] = aa2; lv[i,j] = level
            if level > max_level: max_level = level
    return (torch.from_numpy(nt).to(DEVICE), torch.from_numpy(c1).to(DEVICE),
            torch.from_numpy(c2).to(DEVICE), torch.from_numpy(c3).to(DEVICE),
            torch.from_numpy(a1).to(DEVICE), torch.from_numpy(a2).to(DEVICE),
            torch.from_numpy(lv).to(DEVICE), torch.from_numpy(roots).to(DEVICE),
            max_level)


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("UNIFIED CIC ENGINE: Integration Test")
print("=" * 70)

# Phase 1: Correctness
print("\n--- Phase 1: Correctness ---")
proofs, names, expected = gen_test_cases()
g_nt, g_c1, g_c2, g_c3, g_a1, g_a2, g_lv, g_roots, max_lv = build_batch(proofs)

valid, root_types, result, whnf_steps = engine.cic_engine_pipeline(
    g_nt, g_c1, g_c2, g_c3, g_a1, g_a2, g_lv, g_roots,
    lookup_gpu, const_types_gpu, def_types_gpu, max_lv)

passed = 0
for i in range(len(names)):
    rt = int(root_types[i].item())
    exp = expected[i]
    ws = int(whnf_steps[i].item())
    ok = (rt == exp)
    if ok: passed += 1
    status = "PASS" if ok else "FAIL"
    extra = f" (whnf={ws})" if ws > 0 else ""
    print(f"  [{status}] {names[i]:40s} type={rt:>8d} (exp={exp}){extra}")

print(f"\n  Correctness: {passed}/{len(names)} ({100*passed/len(names):.0f}%)")

# Phase 2: Backward compat (cic_gpu_type_check)
print("\n--- Phase 2: Backward Compatibility ---")
valid2, root_types2, result2 = engine.cic_gpu_type_check(
    g_nt.clone(), g_c1.clone(), g_c2.clone(), g_c3.clone(),
    g_a1.clone(), g_a2.clone(), g_lv.clone(), g_roots.clone(),
    lookup_gpu, const_types_gpu, def_types_gpu, max_lv)

compat_ok = torch.equal(root_types[:len(names)], root_types2[:len(names)])
print(f"  Backward compat: {'PASS' if compat_ok else 'FAIL'}")

# Phase 3: Throughput
print("\n--- Phase 3: Throughput ---")
ev_s = torch.cuda.Event(enable_timing=True)
ev_e = torch.cuda.Event(enable_timing=True)

for bs in [100, 1000, 10000, 100000, 500000, 1000000]:
    batch_proofs = [proofs[i % len(proofs)] for i in range(bs)]
    tensors = build_batch(batch_proofs)
    g_nt, g_c1, g_c2, g_c3, g_a1, g_a2, g_lv, g_roots, max_lv = tensors

    # Warmup
    engine.cic_engine_pipeline(g_nt, g_c1, g_c2, g_c3, g_a1, g_a2, g_lv, g_roots,
                                lookup_gpu, const_types_gpu, def_types_gpu, max_lv)
    torch.cuda.synchronize()

    ev_s.record()
    engine.cic_engine_pipeline(g_nt, g_c1, g_c2, g_c3, g_a1, g_a2, g_lv, g_roots,
                                lookup_gpu, const_types_gpu, def_types_gpu, max_lv)
    ev_e.record(); torch.cuda.synchronize()
    ms = ev_s.elapsed_time(ev_e)

    pps = bs / (ms / 1000) if ms > 0 else 0
    print(f"  {bs:>9,d} proofs: {ms:>10.3f}ms = {pps:>14,.0f} proofs/sec")

    del tensors, g_nt, g_c1, g_c2, g_c3, g_a1, g_a2, g_lv, g_roots
    torch.cuda.empty_cache()

# Phase 4: Real Lean4 theorems
print("\n--- Phase 4: Real Lean4 Theorems ---")
sys.path.insert(0, os.path.dirname(WORKDIR))
from lean4_to_gpu import parse_theorems_v1, flatten_tree_v2, build_gpu_batch

export_path = os.path.join(os.path.dirname(WORKDIR), 'lean4', 'exported_trees.txt')
with open(export_path, encoding='utf-8') as f:
    text = f.read()

theorems = parse_theorems_v1(text)
all_proofs = []
all_names = []

for name, tree in theorems.items():
    nodes, root = flatten_tree_v2(tree, env)
    all_proofs.append((nodes, root))
    all_names.append(name)

batch = build_gpu_batch(all_proofs, env, max_nodes=256)

valid_l, root_types_l, result_l, whnf_l = engine.cic_engine_pipeline(
    batch['node_types'], batch['child1'], batch['child2'], batch['child3'],
    batch['aux1'], batch['aux2'], batch['levels'], batch['roots'],
    batch['lookup'], batch['const_types'], batch['def_values'],
    batch['max_level'])

for i in range(len(all_names)):
    v = int(valid_l[i].item())
    rt = int(root_types_l[i].item())
    ws = int(whnf_l[i].item())
    status = "OK" if v else "FAIL"
    print(f"  [{status}] {all_names[i]:25s} root_type={rt:>10d} whnf_steps={ws}")

print(f"\n  Valid: {int(valid_l.sum())}/{len(all_names)} Lean4 theorems")

# Summary
print(f"\n{'=' * 70}")
print(f"""
UNIFIED CIC ENGINE SUMMARY
============================
  Correctness:   {passed}/{len(names)} test cases
  Backward compat: {'PASS' if compat_ok else 'FAIL'}
  Lean4 theorems: {int(valid_l.sum())}/{len(all_names)} valid

  Architecture:
    Phase 1: WHNF reduction (beta/delta/zeta with de Bruijn substitution)
    Phase 2: Level-by-level type checking
    Phase 3: Result extraction
    All in one kernel module — no need to coordinate 4 separate kernels.
""")
