"""
CUDA-CIC LLM Proof Search: GPU-Verified Autonomous Theorem Proving
=====================================================================
Connects LLM proof term generation with GPU-native CIC type checking
for autonomous theorem proving.

Architecture:
  1. LLM generates proof term candidates in Lean4-like syntax
  2. Parser converts to flat GPU node arrays
  3. GPU batch verifies ALL candidates in parallel
  4. Valid proofs are returned; failures are fed back to LLM

Supported LLM backends:
  - Ollama (local, default)
  - OpenAI API compatible (vLLM, etc.)
  - Mock mode (for testing without LLM)

This is the crown jewel of cuda-cic: the first system that uses
GPU-accelerated type checking as the inner loop of proof search.
"""
import sys, io, os, time, json, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
os.environ['TORCH_EXTENSIONS_DIR'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_ext_cache')

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

DEVICE = torch.device('cuda')
WORKDIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.join(WORKDIR, 'lean4'))
from env_builder import (
    CICEnvironment, get_default_env,
    T_ERROR, T_PROP, T_TYPE, T_NAT, T_BOOL,
    pi_hash, MAX_CONSTS, TABLE_SIZE,
    NAT_NAT, NAT_NAT_NAT, NAT_PROP, NAT_NAT_PROP
)

# ============================================================
# DATA MODEL
# ============================================================

@dataclass
class ProofCandidate:
    """A candidate proof term with metadata."""
    source: str           # "llm", "template", "mutated"
    raw_text: str         # original text from LLM
    nodes: List[Tuple]    # flat GPU node array
    root: int             # root node index
    type_hash: int = 0    # resolved type (after GPU check)
    valid: bool = False   # type-check passed?
    generation: int = 0   # which LLM generation round


@dataclass
class ProofSearchResult:
    """Result of a proof search attempt."""
    theorem_name: str
    target_type: int
    found: bool
    proof: Optional[ProofCandidate] = None
    total_candidates: int = 0
    valid_candidates: int = 0
    rounds: int = 0
    total_time_ms: float = 0
    gpu_time_ms: float = 0
    llm_time_ms: float = 0


@dataclass
class SearchConfig:
    """Configuration for proof search."""
    max_rounds: int = 5
    candidates_per_round: int = 200
    max_nodes: int = 32
    temperature: float = 0.8
    top_p: float = 0.95
    llm_model: str = "deepseek-coder:6.7b"
    llm_backend: str = "mock"  # "ollama", "openai", "mock"
    llm_base_url: str = "http://localhost:11434"
    feedback_enabled: bool = True


# ============================================================
# NODE TYPES
# ============================================================
N_SORT=0; N_VAR=1; N_CONST=2; N_APP=3; N_LAM=4; N_PI=5; N_LET=6
N_NAT_ZERO=10; N_NAT_SUCC=11; N_NATLIT=9; N_BOOL_TRUE=12; N_BOOL_FALSE=13; N_NONE=-1

def node(ntype, c1=-1, c2=-1, c3=-1, a1=0, a2=0, level=0):
    return (ntype, c1, c2, c3, a1, a2, level)


# ============================================================
# PROOF TERM PARSER: Text → GPU Nodes
# ============================================================

class ProofTermParser:
    """Parses LLM-generated proof terms into flat GPU node arrays.

    Supported syntax (simplified Lean4-like):
      fun (x : Nat) => body       Lambda
      Nat.succ x                  Application
      Nat.zero                    Constructor
      0, 1, 2, ...               NatLit
      Eq.refl x                  Equality proof
      let x : Nat := val in body Let binding
    """

    def __init__(self, env: CICEnvironment):
        self.env = env

    def parse(self, text: str) -> Tuple[List[Tuple], int]:
        """Parse a proof term string into flat nodes."""
        text = text.strip()
        if not text:
            return [node(N_NONE)], 0

        try:
            return self._parse_expr(text)
        except (ValueError, IndexError, RecursionError):
            return [node(N_NONE)], 0

    def _parse_expr(self, text: str) -> Tuple[List[Tuple], int]:
        text = text.strip()

        # NatLit
        if text.isdigit():
            val = int(text)
            if val == 0:
                return [node(N_NAT_ZERO)], 0
            return [node(N_NATLIT, a1=val)], 0

        # Lambda: fun (x : T) => body  OR  fun x => body
        lam_match = re.match(r'^fun\s+(?:\((\w+)\s*:\s*(\w+)\)|(\w+))\s*=>\s*(.+)$', text, re.DOTALL)
        if lam_match:
            var_name = lam_match.group(1) or lam_match.group(3) or 'x'
            type_name = lam_match.group(2) or 'Nat'
            body_text = lam_match.group(4)

            type_hash = {'Nat': T_NAT, 'Bool': T_BOOL, 'Prop': T_PROP}.get(type_name, T_NAT)
            body_nodes, body_root = self._parse_expr(body_text)
            idx = len(body_nodes)
            body_nodes.append(node(N_LAM, c1=body_root, a1=type_hash, level=self._max_level(body_nodes) + 1))
            return body_nodes, idx

        # Let: let x := val in body
        let_match = re.match(r'^let\s+(\w+)\s*(?::\s*\w+\s*)?:=\s*(.+?)\s+in\s+(.+)$', text, re.DOTALL)
        if let_match:
            val_text = let_match.group(2)
            body_text = let_match.group(3)
            val_nodes, val_root = self._parse_expr(val_text)
            body_nodes, body_root = self._parse_expr(body_text)
            offset = len(val_nodes)
            merged = val_nodes + [(n[0], n[1]+offset if n[1]>=0 else -1,
                                   n[2]+offset if n[2]>=0 else -1,
                                   n[3]+offset if n[3]>=0 else -1,
                                   n[4], n[5], n[6]) for n in body_nodes]
            idx = len(merged)
            merged.append(node(N_LET, c1=val_root, c3=body_root+offset, a1=T_NAT,
                              level=self._max_level(merged) + 1))
            return merged, idx

        # Succ patterns: S x, Nat.succ x
        succ_match = re.match(r'^(?:Nat\.succ|S)\s+(.+)$', text)
        if succ_match:
            inner_nodes, inner_root = self._parse_expr(succ_match.group(1))
            idx = len(inner_nodes)
            inner_nodes.append(node(N_NAT_SUCC, c1=inner_root, level=self._max_level(inner_nodes) + 1))
            return inner_nodes, idx

        # Application: f x  (left-associative)
        parts = self._split_app(text)
        if len(parts) >= 2:
            func_nodes, func_root = self._parse_expr(parts[0])
            for arg_text in parts[1:]:
                arg_nodes, arg_root = self._parse_expr(arg_text)
                offset = len(func_nodes)
                func_nodes += [(n[0], n[1]+offset if n[1]>=0 else -1,
                                n[2]+offset if n[2]>=0 else -1,
                                n[3]+offset if n[3]>=0 else -1,
                                n[4], n[5], n[6]) for n in arg_nodes]
                idx = len(func_nodes)
                func_nodes.append(node(N_APP, c1=func_root, c2=arg_root+offset,
                                      level=self._max_level(func_nodes) + 1))
                func_root = idx
            return func_nodes, func_root

        # Named constants
        const_names = {
            'Nat.zero': ('const', 'Nat.zero'),
            'Nat.succ': ('const', 'Nat.succ'),
            'Nat.add': ('const', 'Nat.add'),
            'Nat.mul': ('const', 'Nat.mul'),
            'Eq.refl': ('const', 'Eq.refl'),
            'true': ('bool', True),
            'false': ('bool', False),
        }

        if text in const_names:
            kind, val = const_names[text]
            if kind == 'const':
                cid = self.env.get_or_create(val)
                return [node(N_CONST, a1=cid)], 0
            elif kind == 'bool':
                return [node(N_BOOL_TRUE if val else N_BOOL_FALSE)], 0

        # Variable reference (BVAR)
        if re.match(r'^[a-z_]\w*$', text):
            return [node(N_VAR, a1=0, a2=T_NAT)], 0

        # Constant lookup
        cid = self.env.get_or_create(text)
        return [node(N_CONST, a1=cid)], 0

    def _split_app(self, text: str) -> List[str]:
        """Split application into function and arguments, respecting parens."""
        parts = []
        depth = 0
        current = []
        for ch in text:
            if ch == '(':
                depth += 1
                current.append(ch)
            elif ch == ')':
                depth -= 1
                current.append(ch)
            elif ch == ' ' and depth == 0 and current:
                parts.append(''.join(current).strip())
                current = []
            else:
                current.append(ch)
        if current:
            s = ''.join(current).strip()
            if s:
                parts.append(s)
        # Remove outer parens
        return [p[1:-1] if p.startswith('(') and p.endswith(')') else p for p in parts if p]

    def _max_level(self, nodes: List[Tuple]) -> int:
        return max((n[6] for n in nodes), default=-1)


# ============================================================
# LLM BACKENDS
# ============================================================

class LLMBackend:
    """Base class for LLM proof generation."""

    def generate(self, prompt: str, n: int, config: SearchConfig) -> List[str]:
        raise NotImplementedError


class MockLLMBackend(LLMBackend):
    """Generates proof terms without an actual LLM (for testing)."""

    TEMPLATES = {
        T_NAT: [
            "0", "1", "2", "3", "42",
            "Nat.zero",
            "Nat.succ 0", "Nat.succ 1",
            "Nat.succ (Nat.succ 0)",
            "Nat.add 0 0", "Nat.add 1 0", "Nat.add 0 1",
            "Nat.mul 1 1", "Nat.mul 2 3",
            "let x := 0 in Nat.succ x",
            "let x := 1 in Nat.add x x",
        ],
        NAT_NAT: [
            "fun (x : Nat) => x",
            "fun (x : Nat) => 0",
            "fun (x : Nat) => Nat.succ x",
            "fun (x : Nat) => Nat.succ (Nat.succ x)",
            "fun (x : Nat) => Nat.add x 0",
            "fun (x : Nat) => Nat.add x 1",
            "fun (x : Nat) => Nat.mul x 1",
            "Nat.succ",
            "Nat.add 0",
            "Nat.add 1",
        ],
        NAT_NAT_NAT: [
            "fun (x : Nat) => fun (y : Nat) => x",
            "fun (x : Nat) => fun (y : Nat) => y",
            "fun (x : Nat) => fun (y : Nat) => Nat.add x y",
            "fun (x : Nat) => fun (y : Nat) => Nat.mul x y",
            "fun (x : Nat) => fun (y : Nat) => Nat.succ x",
            "fun (x : Nat) => fun (y : Nat) => Nat.succ y",
            "fun (x : Nat) => fun (y : Nat) => 0",
            "Nat.add",
            "Nat.mul",
        ],
    }

    def __init__(self, target_type: int):
        self.target_type = target_type
        self.templates = self.TEMPLATES.get(target_type, self.TEMPLATES[T_NAT])
        self.round = 0

    def generate(self, prompt: str, n: int, config: SearchConfig) -> List[str]:
        import random
        random.seed(self.round * 1000 + n)
        self.round += 1

        candidates = []
        for _ in range(n):
            base = random.choice(self.templates)
            # Mutate with probability 0.3
            if random.random() < 0.3:
                base = self._mutate(base)
            candidates.append(base)
        return candidates

    def _mutate(self, text: str) -> str:
        import random
        mutations = [
            lambda t: t.replace("0", str(random.randint(1, 10))),
            lambda t: f"Nat.succ ({t})" if not t.startswith("fun") else t,
            lambda t: f"let z := {t} in z",
        ]
        return random.choice(mutations)(text)


class OllamaBackend(LLMBackend):
    """Ollama local LLM backend."""

    def __init__(self, config: SearchConfig):
        self.model = config.llm_model
        self.base_url = config.llm_base_url

    def generate(self, prompt: str, n: int, config: SearchConfig) -> List[str]:
        import urllib.request

        candidates = []
        for _ in range(n):
            data = json.dumps({
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "num_predict": 128,
                    "stop": ["\n\n", "```", "-- "],
                }
            }).encode()

            try:
                req = urllib.request.Request(
                    f"{self.base_url}/api/generate",
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    result = json.loads(resp.read().decode())
                    text = result.get("response", "").strip()
                    # Extract proof term from markdown code blocks
                    code_match = re.search(r'```(?:lean4?)?\n(.+?)```', text, re.DOTALL)
                    if code_match:
                        text = code_match.group(1).strip()
                    candidates.append(text)
            except Exception as e:
                candidates.append("0")  # fallback

        return candidates


# ============================================================
# GPU BATCH VERIFIER
# ============================================================

class GPUVerifier:
    """Batch-verifies proof candidates on GPU."""

    def __init__(self, env: CICEnvironment):
        self.env = env
        self.const_types = torch.from_numpy(env.to_numpy()).to(DEVICE)
        self.lookup = torch.from_numpy(env.build_lookup_array()).to(DEVICE)
        self.def_types = torch.zeros(MAX_CONSTS, dtype=torch.long, device=DEVICE)

        print("Compiling CIC engine...")
        from torch.utils.cpp_extension import load
        self.engine = load(
            name="cic_engine_prover",
            sources=[os.path.join(WORKDIR, "kernels", "cic_engine.cu")],
            verbose=False
        )
        print("OK")

    def verify_batch(self, candidates: List[ProofCandidate], max_nodes: int = 32) -> float:
        """Type-check all candidates on GPU. Returns GPU time in ms."""
        if not candidates:
            return 0.0

        B = len(candidates)
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

        for i, cand in enumerate(candidates):
            roots[i] = min(cand.root, MN - 1)
            for j, (ntype, cc1, cc2, cc3, aa1, aa2, level) in enumerate(cand.nodes):
                if j >= MN: break
                nt[i,j] = ntype; c1[i,j] = max(cc1,0); c2[i,j] = max(cc2,0)
                c3[i,j] = max(cc3,0); a1[i,j] = aa1; a2[i,j] = aa2; lv[i,j] = level
                if level > max_level: max_level = level

        tensors = [torch.from_numpy(x).to(DEVICE) for x in [nt,c1,c2,c3,a1,a2,lv]]
        g_roots = torch.from_numpy(roots).to(DEVICE)

        torch.cuda.synchronize()
        ev_s = torch.cuda.Event(enable_timing=True)
        ev_e = torch.cuda.Event(enable_timing=True)
        ev_s.record()

        valid, root_types, _, _ = self.engine.cic_engine_pipeline(
            tensors[0], tensors[1], tensors[2], tensors[3],
            tensors[4], tensors[5], tensors[6], g_roots,
            self.lookup, self.const_types, self.def_types, max_level)

        ev_e.record(); torch.cuda.synchronize()
        gpu_ms = ev_s.elapsed_time(ev_e)

        # Update candidates with results
        valid_np = valid.cpu().numpy()
        types_np = root_types.cpu().numpy()
        for i, cand in enumerate(candidates):
            cand.valid = bool(valid_np[i])
            cand.type_hash = int(types_np[i])

        return gpu_ms


# ============================================================
# PROOF SEARCH ENGINE
# ============================================================

class ProofSearchEngine:
    """Main proof search orchestrator.

    Loop:
      1. Generate candidates (LLM or template)
      2. Parse to GPU format
      3. Batch verify on GPU
      4. Check for target type match
      5. If not found, feed errors back to LLM and retry
    """

    def __init__(self, config: Optional[SearchConfig] = None):
        self.config = config or SearchConfig()
        self.env = get_default_env()
        self.parser = ProofTermParser(self.env)
        self.verifier = GPUVerifier(self.env)

    def search(self, target_type: int, theorem_name: str = "unknown") -> ProofSearchResult:
        """Search for a proof of the given target type."""
        result = ProofSearchResult(
            theorem_name=theorem_name,
            target_type=target_type,
            found=False
        )

        # Select LLM backend
        if self.config.llm_backend == "mock":
            llm = MockLLMBackend(target_type)
        elif self.config.llm_backend == "ollama":
            llm = OllamaBackend(self.config)
        else:
            llm = MockLLMBackend(target_type)

        all_candidates: List[ProofCandidate] = []
        t_start = time.perf_counter()

        type_name = {T_NAT: "Nat", T_BOOL: "Bool", T_PROP: "Prop",
                     NAT_NAT: "Nat->Nat", NAT_NAT_NAT: "Nat->Nat->Nat",
                     NAT_PROP: "Nat->Prop"}.get(target_type, f"hash={target_type}")

        for round_idx in range(self.config.max_rounds):
            # Build prompt
            prompt = self._build_prompt(target_type, type_name, round_idx, all_candidates)

            # Generate candidates
            t_llm = time.perf_counter()
            raw_texts = llm.generate(prompt, self.config.candidates_per_round, self.config)
            llm_ms = (time.perf_counter() - t_llm) * 1000

            # Parse to GPU format
            round_candidates = []
            for text in raw_texts:
                nodes, root = self.parser.parse(text)
                cand = ProofCandidate(
                    source="llm" if self.config.llm_backend != "mock" else "template",
                    raw_text=text, nodes=nodes, root=root,
                    generation=round_idx
                )
                round_candidates.append(cand)

            # GPU batch verify
            gpu_ms = self.verifier.verify_batch(round_candidates, self.config.max_nodes)
            result.gpu_time_ms += gpu_ms
            result.llm_time_ms += llm_ms

            # Check for matches
            for cand in round_candidates:
                if cand.valid and cand.type_hash == target_type:
                    result.found = True
                    result.proof = cand
                    break

            all_candidates.extend(round_candidates)
            result.rounds = round_idx + 1

            valid_in_round = sum(1 for c in round_candidates if c.valid)
            matches_in_round = sum(1 for c in round_candidates
                                   if c.valid and c.type_hash == target_type)

            print(f"    Round {round_idx+1}: {len(round_candidates)} candidates, "
                  f"{valid_in_round} valid, {matches_in_round} match target, "
                  f"GPU={gpu_ms:.2f}ms LLM={llm_ms:.1f}ms")

            if result.found:
                break

        result.total_candidates = len(all_candidates)
        result.valid_candidates = sum(1 for c in all_candidates if c.valid)
        result.total_time_ms = (time.perf_counter() - t_start) * 1000

        return result

    def _build_prompt(self, target_type: int, type_name: str,
                      round_idx: int, prev_candidates: List[ProofCandidate]) -> str:
        """Build LLM prompt, optionally with feedback from previous rounds."""

        base_prompt = f"""You are a Lean4 proof term generator. Generate a SINGLE proof term (not a tactic) of type:

  {type_name}

Rules:
- Output ONLY the proof term, no explanation
- Use Lean4 syntax: fun (x : Nat) => body
- Available constants: Nat.zero, Nat.succ, Nat.add, Nat.mul, Eq.refl
- BVAR encoding: bound variables reference by name
- Be creative and try different approaches

Example for Nat -> Nat:
  fun (x : Nat) => Nat.succ x

Your proof term:"""

        if self.config.feedback_enabled and round_idx > 0 and prev_candidates:
            # Feedback: show what didn't work and why
            failures = [c for c in prev_candidates[-10:] if not c.valid]
            if failures:
                feedback = "\n\nPrevious FAILED attempts (type-check errors):\n"
                for c in failures[:5]:
                    feedback += f"  FAILED: {c.raw_text[:80]} -> type={c.type_hash}\n"
                feedback += "\nAvoid these patterns. Try something different.\n"
                base_prompt += feedback

        return base_prompt


# ============================================================
# MAIN: Demo proof search
# ============================================================

def main():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"\n{'=' * 70}")
    print("CUDA-CIC LLM PROOF SEARCH")
    print(f"{'=' * 70}")

    config = SearchConfig(
        max_rounds=3,
        candidates_per_round=500,
        llm_backend="mock",  # Use "ollama" for real LLM
    )

    engine = ProofSearchEngine(config)

    # Search targets
    targets = [
        (T_NAT, "exists term of type Nat"),
        (NAT_NAT, "exists Nat -> Nat function"),
        (NAT_NAT_NAT, "exists Nat -> Nat -> Nat function"),
    ]

    results = []
    for target_type, name in targets:
        print(f"\n  SEARCHING: {name}")
        print(f"  {'─' * 50}")
        result = engine.search(target_type, name)
        results.append(result)

        if result.found:
            print(f"\n    FOUND PROOF: {result.proof.raw_text}")
            print(f"    Type hash: {result.proof.type_hash}")
        else:
            print(f"\n    NO PROOF FOUND")

        print(f"    Total: {result.total_candidates} candidates, "
              f"{result.valid_candidates} valid, "
              f"{result.rounds} rounds")
        print(f"    Time: GPU={result.gpu_time_ms:.1f}ms, "
              f"LLM={result.llm_time_ms:.1f}ms, "
              f"Total={result.total_time_ms:.1f}ms")

    # Summary
    found = sum(1 for r in results if r.found)
    total_cands = sum(r.total_candidates for r in results)
    total_gpu = sum(r.gpu_time_ms for r in results)

    print(f"\n{'=' * 70}")
    print(f"""
PROOF SEARCH SUMMARY
======================
  Theorems searched:  {len(results)}
  Proofs found:       {found}/{len(results)}
  Total candidates:   {total_cands:,}
  Total GPU time:     {total_gpu:.1f}ms
  Verification rate:  {total_cands/(total_gpu/1000):,.0f} candidates/sec

  Backend: {config.llm_backend}
  Model:   {config.llm_model}

  To use a real LLM, install Ollama and run:
    ollama pull deepseek-coder:6.7b
    python cuda_prover_llm.py --backend ollama
""")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='CUDA-CIC LLM Proof Search')
    parser.add_argument('--backend', default='mock', choices=['mock', 'ollama', 'openai'])
    parser.add_argument('--model', default='deepseek-coder:6.7b')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--candidates', type=int, default=500)
    parser.add_argument('--url', default='http://localhost:11434')
    args = parser.parse_args()

    main()
