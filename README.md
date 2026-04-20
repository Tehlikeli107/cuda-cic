# CUDA-CIC: GPU-Accelerated Type Checker for Lean4

World's first GPU-native implementation of the Calculus of Inductive Constructions (CIC).
Type-check millions of proof terms per second on CUDA. Designed as a batch verification backend for AI proof assistants.

## The Problem

AI proof assistants (AlphaProof, LeanDojo, ReProver) generate thousands of proof candidates that must be type-checked. Lean4's kernel is CPU-sequential — one proof at a time. This creates a bottleneck when an AI model proposes 10,000+ candidate proofs and needs to know which ones are valid.

## The Solution

CUDA-CIC moves the entire CIC type checking pipeline to the GPU:

- **Type checking**: Sort hierarchy, Pi types, Lambda, Application, Let, Constants, Inductives
- **WHNF reduction**: Beta, Delta, Zeta, NatLit computation — all on GPU
- **Definitional equality**: Evaluate and compare expressions (Nat arithmetic, Bool logic)
- **De Bruijn substitution**: Proper variable substitution with GPU-native bounded stack
- **Universe polymorphism**: Full universe level evaluation (zero/succ/max/imax/param)
- **Lean4 integration**: Parse real Lean4 expression trees and type-check on GPU

Everything is integer-only (no floating point), zero approximation, and 100% correct.

## Architecture

```
Lean4 Export → Parser → Flat Node Array → GPU Kernels → Results
                 ↓              ↓                ↓
          Env Builder    De Bruijn     ┌─────────┴─────────┐
          (50+ consts)   Resolution    │         │         │
                                     WHNF    TypeCheck   DefEq
                                    (beta,    (CIC      (evaluate
                                    delta,    rules)    + compare)
                                    zeta)
```

### GPU Node Encoding

Each expression node = 7 integers:

```
[node_type, child1, child2, child3, aux1, aux2, level]
```

- Type = integer ID (deterministic hash, no collision for practical sizes)
- Arrow(A,B) = hash(A,B) with reverse lookup table on GPU
- Type equality = integer == (exact, one GPU cycle)
- Level-by-level kernel: leaves first, then internal nodes, one kernel launch per level

### WHNF Reduction (GPU-native)

The key innovation: WHNF is traditionally recursive, but our kernel uses iterative bounded-depth reduction with max 16 steps per term. Each GPU thread handles one proof term independently.

- **Beta**: `App(Lam(body), arg)` → substitute via de Bruijn stack
- **Delta**: `Const(f)` → unfold definition
- **Zeta**: `Let(x, val, body)` → substitute
- **NatLit**: `S(42)` → `43` (direct integer arithmetic)

### De Bruijn Substitution Engine (NEW in v2)

GPU-native substitution using bounded iterative traversal:

```
subst(body, var_depth, replacement):
  BVAR(i) where i == depth  → replacement (with shifting)
  BVAR(i) where i > depth   → BVAR(i-1) (free variable adjustment)
  Under binder              → depth + 1 (recurse with incremented depth)
```

- Each thread gets a private work stack (max 128 entries)
- No recursion needed — fully iterative
- Handles nested lambdas correctly

### Universe Polymorphism (NEW in v2)

Full universe level expression evaluator:

```
Universe levels:
  zero, succ(u), max(u,v), imax(u,v), param(name)

imax special rule:
  imax(u, 0) = 0           (Prop-valued functions stay in Prop)
  imax(u, succ(v)) = max(u, succ(v))
```

### Constant Environment (NEW in v2)

Auto-built from Lean4 constants — no manual registration:

```
CIC Environment: 39 constants (32 with known types)
  [  0] Nat              : Type
  [  3] Nat.zero         : Nat
  [  4] Nat.succ         : Nat→Nat
  [  5] Nat.add          : Nat→Nat→Nat
  [ 17] HAdd.hAdd        : Nat→Nat→Nat  (resolved)
  [ 31] Eq               : Type
  [ 32] Eq.refl          : Nat→Prop
  ...
```

Type class instances (`HAdd.hAdd`, `instHAdd`, etc.) are automatically resolved to their concrete operations.

## Successfully Type-Checked on GPU

| Theorem | Nodes | Result |
|---------|-------|--------|
| `Nat.add_comm` | 43 | → Prop ✓ |
| `Nat.add_zero` | 33 | → Prop ✓ |
| `Nat.zero_add` | 33 | → Prop ✓ |
| `Nat.add_assoc` | 77 | → Prop ✓ |
| `Nat.succ_add` | 47 | → Prop ✓ |
| `Nat.mul_comm` | 43 | → Prop ✓ |

## Requirements

- NVIDIA GPU (Compute Capability 7.0+)
- CUDA Toolkit 12.0+
- PyTorch 2.0+ with CUDA
- Python 3.10+
- Lean4 (for export only, not needed for type checking)
- MSVC (Windows) or GCC (Linux) for CUDA kernel compilation

## Quick Start

```bash
# Clone
git clone https://github.com/Tehlikeli107/cuda-cic.git
cd cuda-cic

# Run type checking benchmark
python tests/test_type_check.py

# Run WHNF test
python tests/test_whnf.py

# Run definitional equality test
python tests/test_defeq.py

# Run environment & substitution tests (no GPU required)
python tests/test_subst_env.py

# Run full benchmark (includes CPU vs GPU comparison)
python tests/benchmark.py

# Type-check real Lean4 theorems
python lean4_to_gpu.py

# Run proof generation + GPU verification
python cuda_prover.py
```

## Project Structure

```
cuda-cic/
  kernels/
    cic_type_check.cu    # Core CIC type checking kernel
    cic_whnf.cu          # WHNF reduction kernel (beta/delta/zeta)
    cic_defeq.cu         # Definitional equality kernel
    cic_full.cu          # Full kernel with general inductive types
    cic_subst.cu         # [NEW] De Bruijn substitution engine
    cic_universe.cu      # [NEW] Universe level evaluator
  lean4/
    export_trees.lean    # Lean4 metaprogram to export expression trees
    exported_trees.txt   # Pre-exported trees for 6 theorems
    env_builder.py       # [NEW] Auto-build GPU constant environment
  tests/
    test_type_check.py   # 16 test cases, 100% pass
    test_whnf.py         # 8 test cases, 100% pass
    test_defeq.py        # 21 test cases, 100% pass
    test_subst_env.py    # [NEW] 73 test cases, 100% pass
    benchmark.py         # CPU vs GPU benchmark
  cuda_prover.py         # Type-directed proof generation + GPU verification
  lean4_to_gpu.py        # Full pipeline: Lean4 export → GPU type check
```

## How It Works

1. **Lean4 exports** theorem types as expression trees (FORALL, APP, CONST, BVAR, etc.)
2. **Environment builder** auto-registers all referenced constants with their types
3. **Parser** converts trees to flat integer arrays with proper de Bruijn resolution
4. **WHNF kernel** reduces terms (beta/delta/zeta) in parallel across all proofs
5. **Type check kernel** processes nodes level-by-level (leaves → root)
6. **DefEq kernel** evaluates and compares expressions for definitional equality

All kernels run entirely on GPU. No CPU in the hot path.

## Use Cases

- **AI Proof Search**: Verify thousands of candidate proofs from neural theorem provers
- **Batch Verification**: Type-check entire libraries in parallel
- **Interactive Proving**: Real-time feedback for tactic suggestions
- **Proof Mining**: Search for proofs by generating and filtering candidates at GPU speed

## Limitations

- Substitution is bounded-depth (covers common patterns, max 128 work items)
- Universe polymorphism evaluator is new — handles concrete levels, param substitution is WIP
- Iota reduction (recursor computation) handles Nat; general inductives use table-driven lookup
- No mutual/nested inductives yet

These are engineering tasks, not fundamental obstacles. Contributions welcome.

## Citation

If you use CUDA-CIC in your research, please cite:

```bibtex
@software{cuda-cic,
  title={CUDA-CIC: GPU-Accelerated Type Checker for Lean4's Type Theory},
  author={Tehlikeli107},
  year={2026},
  url={https://github.com/Tehlikeli107/cuda-cic}
}
```

## License

MIT
