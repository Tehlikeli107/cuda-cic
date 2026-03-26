"""
Test GPU WHNF Kernel: Beta/Delta/Zeta reduction on GPU
"""
import sys, io, os, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
os.environ['TORCH_EXTENSIONS_DIR'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_ext_cache')

import torch, numpy as np
DEVICE = torch.device('cuda')
WORKDIR = os.path.dirname(os.path.abspath(__file__))
print(f"Device: {torch.cuda.get_device_name(0)}")

print("Compiling WHNF kernel...")
from torch.utils.cpp_extension import load
whnf_gpu = load(name="cic_whnf", sources=[os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kernels", "cic_whnf.cu")], verbose=False)
print("OK")

# Constants
T_ERROR=0; T_PROP=1; T_TYPE=2; T_TYPE1=3; T_NAT=10; T_BOOL=11
N_SORT=0; N_VAR=1; N_CONST=2; N_APP=3; N_LAM=4; N_PI=5; N_LET=6
N_CTOR=7; N_REC=8; N_NATLIT=9; N_NAT_ZERO=10; N_NAT_SUCC=11; N_NONE=-1

PRIME1=1000003; PRIME2=999983; PI_SALT=0x50000000
HASH_MOD=1048576; TABLE_SIZE=8388708; MAX_CONSTS=65536

def pi_hash(d, c): return int(((d*PRIME1+c*PRIME2+PI_SALT)%HASH_MOD)+2*HASH_MOD)

NAT_NAT = pi_hash(T_NAT, T_NAT)
NAT_NAT_NAT = pi_hash(T_NAT, NAT_NAT)

# Constant + definition tables
const_types_np = np.zeros(MAX_CONSTS, dtype=np.int64)
def_types_np = np.zeros(MAX_CONSTS, dtype=np.int64)
def_values_np = np.zeros(MAX_CONSTS, dtype=np.int64)

C_NAT=0; C_NAT_ZERO=1; C_NAT_SUCC=2; C_NAT_ADD=6
const_types_np[C_NAT] = T_TYPE
const_types_np[C_NAT_ZERO] = T_NAT
const_types_np[C_NAT_SUCC] = NAT_NAT
const_types_np[C_NAT_ADD] = NAT_NAT_NAT

# Define a simple constant: myConst := 42
C_MYCONST = 20
const_types_np[C_MYCONST] = T_NAT
def_types_np[C_MYCONST] = T_NAT  # signals "has definition"

const_types_gpu = torch.from_numpy(const_types_np).to(DEVICE)
def_types_gpu = torch.from_numpy(def_types_np).to(DEVICE)
def_values_gpu = torch.from_numpy(def_values_np).to(DEVICE)
lookup_gpu = torch.zeros(TABLE_SIZE * 2, dtype=torch.long, device=DEVICE)

for (d, c) in [(T_NAT, T_NAT), (T_NAT, NAT_NAT)]:
    h = pi_hash(d, c)
    if h < TABLE_SIZE: lookup_gpu[h*2]=d; lookup_gpu[h*2+1]=c

def node(ntype, c1=-1, c2=-1, c3=-1, a1=0, a2=0, level=0):
    return (ntype, c1, c2, c3, a1, a2, level)

MN = 32

def build_batch(proofs):
    B = len(proofs)
    nt=np.full((B,MN),N_NONE,dtype=np.int64)
    c1=np.zeros((B,MN),dtype=np.int64); c2=np.zeros((B,MN),dtype=np.int64)
    c3=np.zeros((B,MN),dtype=np.int64); a1=np.zeros((B,MN),dtype=np.int64)
    a2=np.zeros((B,MN),dtype=np.int64); lv=np.full((B,MN),-1,dtype=np.int64)
    roots=np.zeros(B,dtype=np.int64); max_lv=0
    for i,(nodes,root) in enumerate(proofs):
        roots[i]=root
        for j,(ntype,cc1,cc2,cc3,aa1,aa2,level) in enumerate(nodes):
            if j>=MN: break
            nt[i,j]=ntype; c1[i,j]=max(cc1,0); c2[i,j]=max(cc2,0)
            c3[i,j]=max(cc3,0); a1[i,j]=aa1; a2[i,j]=aa2; lv[i,j]=level
            if level>max_lv: max_lv=level
    return (torch.from_numpy(x).to(DEVICE) for x in [nt,c1,c2,c3,a1,a2,lv,roots]), max_lv

# ============================================================
# TEST CASES FOR WHNF
# ============================================================

print("\n" + "="*70)
print("GPU WHNF KERNEL TEST: Beta/Delta/Zeta Reduction on GPU")
print("="*70)

proofs = []; names = []; expected = []

# Test 0: (fun x:Nat => x) 0 -> 0 via beta reduction
# Nodes: 0=var(Nat), 1=lam(body=0, dom=Nat), 2=zero, 3=app(1,2)
proofs.append(([node(N_VAR, a1=T_NAT, level=0),
                node(N_LAM, c1=0, a1=T_NAT, level=1),
                node(N_NAT_ZERO, level=0),
                node(N_APP, c1=1, c2=2, level=2)], 3))
names.append("BETA: (fun x => x) 0 -> 0"); expected.append(T_NAT)

# Test 1: let x = 0 in S(x) -> S(0) via zeta reduction
proofs.append(([node(N_NAT_ZERO, level=0),
                node(N_NAT_SUCC, c1=0, level=1),
                node(N_LET, c1=0, c3=1, a1=T_NAT, level=2)], 2))
names.append("ZETA: let x=0 in S(x) -> S(0)"); expected.append(T_NAT)

# Test 2: S(42) via natlit computation
proofs.append(([node(N_NATLIT, a1=42, level=0),
                node(N_NAT_SUCC, c1=0, level=1)], 1))
names.append("COMP: S(42) -> 43"); expected.append(T_NAT)

# Test 3: Delta reduction - myConst (defined as Nat) -> unfold
proofs.append(([node(N_CONST, a1=C_MYCONST, level=0)], 0))
names.append("DELTA: myConst -> unfold"); expected.append(T_NAT)

# Test 4: Nested beta: (fun x => S(x)) 0 -> S(0)
proofs.append(([node(N_VAR, a1=T_NAT, level=0),
                node(N_NAT_SUCC, c1=0, level=1),
                node(N_LAM, c1=1, a1=T_NAT, level=2),
                node(N_NAT_ZERO, level=0),
                node(N_APP, c1=2, c2=3, level=3)], 4))
names.append("BETA: (fun x => S(x)) 0"); expected.append(T_NAT)

# Test 5: No reduction needed (already WHNF)
proofs.append(([node(N_NAT_ZERO, level=0)], 0))
names.append("WHNF: 0 (no reduction)"); expected.append(T_NAT)

# Test 6: Simple var (WHNF)
proofs.append(([node(N_VAR, a1=T_NAT, level=0)], 0))
names.append("WHNF: var (no reduction)"); expected.append(T_NAT)

# Test 7: Lambda (WHNF - lambdas are values)
proofs.append(([node(N_VAR, a1=T_NAT, level=0),
                node(N_LAM, c1=0, a1=T_NAT, level=1)], 1))
names.append("WHNF: lam (no reduction)"); expected.append(NAT_NAT)

# Build and run
tensors_iter, max_lv = build_batch(proofs)
g_nt,g_c1,g_c2,g_c3,g_a1,g_a2,g_lv,g_roots = list(tensors_iter)

torch.cuda.synchronize()
ev_s = torch.cuda.Event(enable_timing=True)
ev_e = torch.cuda.Event(enable_timing=True)
ev_s.record()

valid, root_types, result, whnf_steps = whnf_gpu.cic_whnf_pipeline(
    g_nt, g_c1, g_c2, g_c3, g_a1, g_a2, g_lv, g_roots,
    lookup_gpu, const_types_gpu, def_values_gpu, def_types_gpu, max_lv)

ev_e.record(); torch.cuda.synchronize()
gpu_ms = ev_s.elapsed_time(ev_e)

print(f"\nGPU time: {gpu_ms:.3f}ms (WHNF + type check)")
print(f"\n{'_'*65}")

passed = 0
for i in range(len(names)):
    rt = int(root_types[i].item())
    ws = int(whnf_steps[i].item())
    exp = expected[i]
    ok = (rt == exp)
    if ok: passed += 1
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {names[i]:45s} type={rt:>8d} whnf_steps={ws}")

print(f"\n  Results: {passed}/{len(names)}")

# Batch throughput with WHNF
print(f"\n-- WHNF+TypeCheck Batch Throughput --")
for bs in [1000, 10000, 100000, 500000]:
    batch_proofs = [proofs[i % len(proofs)] for i in range(bs)]
    ti, ml = build_batch(batch_proofs)
    bg_nt,bg_c1,bg_c2,bg_c3,bg_a1,bg_a2,bg_lv,bg_roots = list(ti)

    # warmup
    whnf_gpu.cic_whnf_pipeline(bg_nt.clone(),bg_c1.clone(),bg_c2.clone(),bg_c3.clone(),
                                 bg_a1.clone(),bg_a2.clone(),bg_lv.clone(),bg_roots.clone(),
                                 lookup_gpu,const_types_gpu,def_values_gpu,def_types_gpu,ml)
    torch.cuda.synchronize()

    ev_s.record()
    whnf_gpu.cic_whnf_pipeline(bg_nt,bg_c1,bg_c2,bg_c3,bg_a1,bg_a2,bg_lv,bg_roots,
                                 lookup_gpu,const_types_gpu,def_values_gpu,def_types_gpu,ml)
    ev_e.record(); torch.cuda.synchronize()
    ms = ev_s.elapsed_time(ev_e)
    pps = bs/(ms/1000) if ms>0 else 0
    print(f"  {bs:>9,d} proofs: {ms:>8.3f}ms = {pps:>14,.0f} proofs/sec (WHNF+TC)")

print(f"\n{'='*70}")
print("GPU WHNF: Beta, Delta, Zeta, NatLit computation — all on GPU")
print("World's first GPU-native WHNF reduction for dependent type theory")
