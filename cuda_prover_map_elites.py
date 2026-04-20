import os
import sys
import time
import math
import numpy as np
import torch
from typing import List, Dict, Tuple, Any

# Ensure CUDA is available
if not torch.cuda.is_available():
    print("FATAL: CUDA not found.")
    sys.exit(1)

DEVICE = torch.device("cuda:0")
WORKDIR = os.path.dirname(os.path.abspath(__file__))

# Import environment builder
sys.path.insert(0, os.path.join(WORKDIR, 'lean4'))
from env_builder import (
    CICEnvironment, get_default_env,
    MAX_CONSTS
)

from lean4_to_gpu import (
    N_NONE, N_SORT, N_CONST, N_VAR, N_APP, N_LAM, N_PI, N_LET, 
    N_CTOR, N_REC, N_NATLIT, N_STRLIT, N_MVAR
)

# ============================================================
# THE HARDWARE-NATIVE ARTIFICIAL MATHEMATICIAN (MAP-ELITES)
# ============================================================

class GPUEvolutionEngine:
    def __init__(self, env: CICEnvironment):
        self.env = env
        self.const_types = torch.from_numpy(env.to_numpy()).to(DEVICE)
        self.lookup = torch.from_numpy(env.build_lookup_array()).to(DEVICE)
        self.def_types = torch.zeros(MAX_CONSTS, dtype=torch.long, device=DEVICE)
        
        self.ctor_tags = torch.full((MAX_CONSTS,), -1, dtype=torch.long, device=DEVICE)
        self.rec_rules = torch.zeros(MAX_CONSTS, dtype=torch.long, device=DEVICE)
        
        for cid, tag in env.id_to_tag.items():
            if 0 <= cid < MAX_CONSTS:
                self.ctor_tags[cid] = tag
                
        for cid, rule in env.id_to_rule.items():
            if 0 <= cid < MAX_CONSTS:
                self.rec_rules[cid] = rule

        print("Compiling Evolutionary Engine (cic_engine.cu)...")
        from torch.utils.cpp_extension import load
        self.engine = load(
            name="cic_engine_map_elites",
            sources=[os.path.join(WORKDIR, "kernels", "cic_engine.cu")],
            verbose=False
        )
        print("Engine compiled and loaded successfully.")

        # Archive state
        self.novel_types_found = set()
        self.archive_roots_cpu = []
        
        # We start with empty archive to force random initial population
        self.archive_tensor = torch.zeros(1, dtype=torch.long, device=DEVICE)
        
    def run_generation(self, batch_size: int, max_nodes: int, gen: int) -> Tuple[int, int, float]:
        """Runs a single mutation and evaluation generation on the GPU."""
        B = batch_size
        MN = max_nodes

        # Pre-allocate blank AST tensors on GPU for the new generation
        nt = torch.full((B, MN), N_NONE, dtype=torch.long, device=DEVICE)
        c1 = torch.full((B, MN), -1, dtype=torch.long, device=DEVICE)
        c2 = torch.full((B, MN), -1, dtype=torch.long, device=DEVICE)
        c3 = torch.full((B, MN), -1, dtype=torch.long, device=DEVICE)
        a1 = torch.zeros((B, MN), dtype=torch.long, device=DEVICE)
        a2 = torch.zeros((B, MN), dtype=torch.long, device=DEVICE)
        lv = torch.zeros((B, MN), dtype=torch.long, device=DEVICE)
        roots = torch.full((B,), -1, dtype=torch.long, device=DEVICE)
        
        # Load pre-allocated foundations
        nt[:, 0] = N_SORT; a1[:, 0] = 0 # Prop
        nt[:, 1] = N_SORT; a1[:, 1] = 1 # Type
        nt[:, 2] = N_CONST; a1[:, 2] = self.env.get_or_create("Nat")
        nt[:, 3] = N_CONST; a1[:, 3] = self.env.get_or_create("Bool")
        nt[:, 4] = N_CONST; a1[:, 4] = self.env.get_or_create("String")

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        # 1. GPU MUTATION (Crossover, Add Random Nodes)
        # 2. GPU TYPE CHECKING
        valid, root_types, _, _ = self.engine.cic_engine_mutate(
            nt, c1, c2, c3, a1, a2, lv, roots,
            self.lookup, self.const_types, self.def_types,
            self.ctor_tags, self.rec_rules,
            self.archive_tensor, len(self.archive_roots_cpu), 
            10, gen + int(t0 * 1000) # seed
        )
        
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        gpu_time = t1 - t0
        
        # 3. PHENOTYPE EVALUATION (CPU side novelty check)
        valid_cpu = valid.cpu().numpy()
        types_cpu = root_types.cpu().numpy()
        
        new_discoveries = 0
        for i in range(B):
            if valid_cpu[i] == 1:
                t_hash = types_cpu[i]
                if t_hash != 0 and t_hash not in self.novel_types_found:
                    self.novel_types_found.add(t_hash)
                    new_discoveries += 1
                    
                    # Add to archive
                    self.archive_roots_cpu.append(0) # Simplified for now
        
        # Update GPU archive tensor if we have new elites
        if new_discoveries > 0 and len(self.archive_roots_cpu) > 0:
            self.archive_tensor = torch.tensor(self.archive_roots_cpu, dtype=torch.long, device=DEVICE)
            
        return len(self.novel_types_found), new_discoveries, gpu_time

def main():
    print("=" * 60)
    print(" CUDA-CIC: THE HARDWARE-NATIVE ARTIFICIAL MATHEMATICIAN ")
    print("=" * 60)
    print(f"Device: {torch.cuda.get_device_name(0)}\n")

    env = get_default_env()
    env.load_from_export(os.path.join(WORKDIR, 'lean4', 'exported_trees.txt'))
    
    print(f"Environment loaded: {env.summary().splitlines()[0]}")
    
    evo = GPUEvolutionEngine(env)
    
    # Evolutionary Hyperparameters
    BATCH_SIZE = 100000   # B=100K ASTs per generation
    MAX_NODES = 32        # Small MN for blazing fast generations
    
    print(f"\n[STARTING EVOLUTIONARY LOOP]")
    print(f"Population: {BATCH_SIZE:,}  |  Max Depth: {MAX_NODES}")
    print("-" * 60)
    
    gen = 0
    total_time = 0.0
    total_ast_checked = 0
    
    try:
        while True:
            gen += 1
            total_elites, novel_in_gen, gpu_time = evo.run_generation(BATCH_SIZE, MAX_NODES, gen)
            
            total_time += gpu_time
            total_ast_checked += BATCH_SIZE
            
            pps = BATCH_SIZE / gpu_time if gpu_time > 0 else 0
            
            # Dashboard Output
            if gen % 10 == 0 or novel_in_gen > 0:
                print(f"Gen {gen:5d} | "
                      f"Elites (Theorems): {total_elites:4d} (+{novel_in_gen}) | "
                      f"Speed: {pps/1e6:.2f} M/sec | "
                      f"GPU: {gpu_time*1000:.1f}ms")
                      
            # Prevent infinite crazy spew in test mode
            if gen >= 200:
                break
                
    except KeyboardInterrupt:
        print("\nEvolution halted by user.")
        
    print("-" * 60)
    print("FINAL ARCHIVE STATISTICS")
    print(f"Total Generations : {gen:,}")
    print(f"ASTs Evaluated    : {total_ast_checked:,}")
    print(f"Theorems Discovered: {len(evo.novel_types_found)}")
    print(f"Average Speed     : {(total_ast_checked / total_time) / 1e6:.2f} M/sec")

if __name__ == '__main__':
    main()