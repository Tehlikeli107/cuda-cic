import Lean
open Lean

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
  | .const n _ =>
    IO.println s!"{pad}CONST {n}"
  | .bvar i =>
    IO.println s!"{pad}BVAR {i}"
  | .sort u =>
    IO.println s!"{pad}SORT {u}"
  | .lam n t b _ =>
    IO.println s!"{pad}LAM {n}"
    showExpr t (depth + 1)
    showExpr b (depth + 1)
  | .lit (.natVal v) =>
    IO.println s!"{pad}NATLIT {v}"
  | _ =>
    IO.println s!"{pad}OTHER"

def exportThm (env : Environment) (name : Name) : IO Unit := do
  match env.find? name with
  | some (.thmInfo tv) =>
    IO.println s!"=== THEOREM {name} ==="
    showExpr tv.type 0
    IO.println "---"
  | _ => IO.println s!"{name}: not found"

#eval show Lean.Elab.Command.CommandElabM _ from do
  let env <- Lean.getEnv
  exportThm env `Nat.add_comm
  exportThm env `Nat.add_zero
  exportThm env `Nat.zero_add
  exportThm env `Nat.add_assoc
  exportThm env `Nat.succ_add
  exportThm env `Nat.mul_comm
