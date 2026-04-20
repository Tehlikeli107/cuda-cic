import Lean
open Lean

-- Extended Expression Exporter: exports both types and proof terms
-- Also exports constant types for all referenced constants

def showExpr (e : Expr) (depth : Nat) : IO Unit := do
  let pad := String.mk (List.replicate (depth * 2) ' ')
  match e with
  | .forallE n t b _ =>
    IO.println s!"{pad}FORALL {n}"
    showExpr t (depth + 1)
    showExpr b (depth + 1)
  | .app f a =>
    IO.println s!"{pad}APP"
    showExpr f (depth + 1)
    showExpr a (depth + 1)
  | .const n us =>
    -- Export universe level instantiations
    if us.isEmpty then
      IO.println s!"{pad}CONST {n}"
    else
      IO.println s!"{pad}CONST {n}"
      for u in us do
        IO.println s!"{pad}  ULEVEL {u}"
  | .bvar i =>
    IO.println s!"{pad}BVAR {i}"
  | .fvar id =>
    IO.println s!"{pad}FVAR {id.name}"
  | .sort u =>
    IO.println s!"{pad}SORT {u}"
  | .lam n t b _ =>
    IO.println s!"{pad}LAM {n}"
    showExpr t (depth + 1)
    showExpr b (depth + 1)
  | .letE n t v b _ =>
    IO.println s!"{pad}LET {n}"
    showExpr t (depth + 1)
    showExpr v (depth + 1)
    showExpr b (depth + 1)
  | .lit (.natVal v) =>
    IO.println s!"{pad}NATLIT {v}"
  | .lit (.strVal s) =>
    IO.println s!"{pad}STRLIT {s}"
  | .mdata _ e =>
    showExpr e depth  -- unwrap metadata
  | .proj typeName idx struct =>
    IO.println s!"{pad}PROJ {typeName} {idx}"
    showExpr struct (depth + 1)
  | .mvar id =>
    IO.println s!"{pad}MVAR {id.name}"

-- Collect all constant names referenced in an expression
partial def collectConsts (e : Expr) (acc : Std.HashSet Name := {}) : Std.HashSet Name :=
  match e with
  | .const n _ => acc.insert n
  | .app f a => collectConsts a (collectConsts f acc)
  | .forallE _ t b _ => collectConsts b (collectConsts t acc)
  | .lam _ t b _ => collectConsts b (collectConsts t acc)
  | .letE _ t v b _ => collectConsts b (collectConsts v (collectConsts t acc))
  | .mdata _ e => collectConsts e acc
  | .proj _ _ s => collectConsts s acc
  | _ => acc

-- Export a theorem with both type and value (proof term)
def exportThmFull (env : Environment) (name : Name) : IO Unit := do
  match env.find? name with
  | some (.thmInfo tv) =>
    IO.println s!"=== THEOREM {name} ==="
    IO.println "--- TYPE ---"
    showExpr tv.type 0
    IO.println "--- PROOF ---"
    showExpr tv.value 0
    IO.println "--- END ---"
  | some (.defnInfo dv) =>
    IO.println s!"=== DEFINITION {name} ==="
    IO.println "--- TYPE ---"
    showExpr dv.type 0
    IO.println "--- VALUE ---"
    showExpr dv.value 0
    IO.println "--- END ---"
  | some (.axiomInfo av) =>
    IO.println s!"=== AXIOM {name} ==="
    IO.println "--- TYPE ---"
    showExpr av.type 0
    IO.println "--- END ---"
  | some (.ctorInfo cv) =>
    IO.println s!"=== CONSTRUCTOR {name} ==="
    IO.println "--- TYPE ---"
    showExpr cv.type 0
    IO.println "--- END ---"
  | some (.recInfo rv) =>
    IO.println s!"=== RECURSOR {name} ==="
    IO.println "--- TYPE ---"
    showExpr rv.type 0
    IO.println "--- END ---"
  | some (.inductInfo iv) =>
    IO.println s!"=== INDUCTIVE {name} ==="
    IO.println "--- TYPE ---"
    showExpr iv.type 0
    IO.println s!"--- CTORS {iv.ctors.length} ---"
    for ctor in iv.ctors do
      IO.println s!"  CTOR {ctor}"
    IO.println "--- END ---"
  | _ => IO.println s!"{name}: not found or unsupported"

-- Export constant type info
def exportConstType (env : Environment) (name : Name) : IO Unit := do
  match env.find? name with
  | some ci =>
    IO.println s!"CONST_TYPE {name}"
    showExpr ci.type 0
    match ci with
    | .defnInfo dv =>
      IO.println "  HAS_VALUE true"
    | _ =>
      IO.println "  HAS_VALUE false"
    IO.println "---"
  | none =>
    IO.println s!"CONST_TYPE {name} NOT_FOUND"
    IO.println "---"


#eval show Lean.Elab.Command.CommandElabM _ from do
  let env ← Lean.getEnv

  -- Header
  IO.println "=== CUDA-CIC LEAN4 EXPORT v2 ==="
  IO.println ""

  -- === SECTION 1: Theorem Types + Proof Terms ===
  IO.println "=== SECTION: THEOREMS ==="

  -- Nat arithmetic
  exportThmFull env `Nat.add_comm
  exportThmFull env `Nat.add_zero
  exportThmFull env `Nat.zero_add
  exportThmFull env `Nat.add_assoc
  exportThmFull env `Nat.succ_add
  exportThmFull env `Nat.mul_comm
  exportThmFull env `Nat.mul_one
  exportThmFull env `Nat.one_mul
  exportThmFull env `Nat.add_left_cancel
  exportThmFull env `Nat.mul_assoc

  -- Simple propositions
  exportThmFull env `Eq.symm
  exportThmFull env `Eq.trans

  IO.println ""

  -- === SECTION 2: Core Constant Types ===
  IO.println "=== SECTION: CONSTANTS ==="

  -- Types
  exportConstType env `Nat
  exportConstType env `Bool
  exportConstType env `Prop
  exportConstType env `Eq

  -- Nat constructors and ops
  exportConstType env `Nat.zero
  exportConstType env `Nat.succ
  exportConstType env `Nat.add
  exportConstType env `Nat.mul
  exportConstType env `Nat.sub
  exportConstType env `Nat.rec

  -- Bool
  exportConstType env `Bool.true
  exportConstType env `Bool.false

  -- HAdd type class machinery
  exportConstType env `HAdd.hAdd
  exportConstType env `Add.add
  exportConstType env `instHAdd
  exportConstType env `instAddNat

  -- HMul
  exportConstType env `HMul.hMul
  exportConstType env `Mul.mul
  exportConstType env `instHMul
  exportConstType env `instMulNat

  -- OfNat
  exportConstType env `OfNat.ofNat
  exportConstType env `instOfNatNat

  -- Eq
  exportConstType env `Eq.refl
  exportConstType env `Eq.symm
  exportConstType env `Eq.trans

  -- Inductive types
  exportThmFull env `Nat
  exportThmFull env `Bool
  exportThmFull env `List
  exportThmFull env `String
  exportThmFull env `Char

  IO.println "=== END EXPORT ==="
