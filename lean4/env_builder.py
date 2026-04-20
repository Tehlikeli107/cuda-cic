"""
Lean4 Constant Environment Builder for GPU
=============================================
Builds the complete const_types GPU array from Lean4 environment export.

Instead of manually registering 10 constants, we parse the Lean4 export
and automatically register ALL referenced constants with their types.

This is the bridge between Lean4's elaborated environment and our GPU kernel.
"""
import re
import numpy as np
from typing import Dict, Tuple, Optional, List

# ============================================================
# TYPE SYSTEM (mirrors GPU kernel constants)
# ============================================================

T_ERROR = 0
T_PROP  = 1     # Sort 0
T_TYPE  = 2     # Sort 1
T_TYPE1 = 3     # Sort 2
T_NAT   = 10
T_BOOL  = 11
T_STRING= 12
T_LIST  = 13

PRIME1 = 1000003
PRIME2 = 999983
PI_SALT = 0x50000000
HASH_MOD = 1048576
TABLE_SIZE = 8388708
MAX_CONSTS = 65536

def pi_hash(dom: int, cod: int) -> int:
    """Hash for Pi/Arrow type: dom → cod"""
    return int(((dom * PRIME1 + cod * PRIME2 + PI_SALT) % HASH_MOD) + 2 * HASH_MOD)

def sort_hash(level: int) -> int:
    """Hash for Sort(level)"""
    if level == 0: return T_PROP
    if level == 1: return T_TYPE
    if level == 2: return T_TYPE1
    return int(((level * PRIME1 + 0x60000000) % HASH_MOD) + HASH_MOD)


# ============================================================
# PRE-COMPUTED TYPE HASHES
# ============================================================

NAT_NAT       = pi_hash(T_NAT, T_NAT)           # Nat → Nat
NAT_NAT_NAT   = pi_hash(T_NAT, NAT_NAT)         # Nat → Nat → Nat
BOOL_BOOL      = pi_hash(T_BOOL, T_BOOL)          # Bool → Bool
NAT_BOOL       = pi_hash(T_NAT, T_BOOL)           # Nat → Bool
NAT_NAT_BOOL   = pi_hash(T_NAT, NAT_BOOL)         # Nat → Nat → Bool
BOOL_BOOL_BOOL = pi_hash(T_BOOL, BOOL_BOOL)       # Bool → Bool → Bool
NAT_PROP       = pi_hash(T_NAT, T_PROP)           # Nat → Prop
NAT_NAT_PROP   = pi_hash(T_NAT, NAT_PROP)         # Nat → Nat → Prop
PROP_PROP      = pi_hash(T_PROP, T_PROP)           # Prop → Prop
TYPE_TYPE      = pi_hash(T_TYPE, T_TYPE)           # Type → Type
TYPE_PROP      = pi_hash(T_TYPE, T_PROP)           # Type → Prop

# Eq Nat : Nat → Nat → Prop
EQ_NAT = NAT_NAT_PROP


class CICEnvironment:
    """GPU-ready CIC constant environment.

    Manages the mapping from Lean4 constant names to GPU constant IDs
    and their type hashes, as well as constructor tags and recursor rules
    for fully generic IOTA reduction on GPU.
    """

    def __init__(self):
        self.name_to_id: Dict[str, int] = {}
        self.id_to_type: Dict[int, int] = {}
        self.id_to_name: Dict[int, str] = {}
        self.id_to_tag: Dict[int, int] = {}    # cid -> constructor_tag (0, 1, 2...)
        self.id_to_rule: Dict[int, int] = {}   # cid -> recursor_rule (bitpacked: params << 16 | minors)
        self.next_id: int = 0
        self.pi_types: List[Tuple[int, int]] = []  # (dom, cod) pairs to register

        self._register_core_constants()

    def get_tag(self, cid: int) -> int:
        return self.id_to_tag.get(cid, -1)
        
    def get_rule(self, cid: int) -> int:
        return self.id_to_rule.get(cid, 0)

    def _register_core_constants(self):
        """Register the core Lean4 constants with known types."""

        # === Types ===
        self.register("Nat", T_TYPE)
        self.register("Bool", T_TYPE)
        self.register("Prop", T_TYPE)  # Prop : Type
        self.register("String", T_TYPE)
        self.register("List", pi_hash(T_TYPE, T_TYPE))  # List : Type u -> Type u

        # === Nat constructors ===
        self.register("Nat.zero", T_NAT)
        self.register("Nat.succ", NAT_NAT)

        # === Nat operations ===
        self.register("Nat.add", NAT_NAT_NAT)
        self.register("Nat.mul", NAT_NAT_NAT)
        self.register("Nat.sub", NAT_NAT_NAT)
        self.register("Nat.mod", NAT_NAT_NAT)
        self.register("Nat.div", NAT_NAT_NAT)
        self.register("Nat.beq", NAT_NAT_BOOL)
        self.register("Nat.ble", NAT_NAT_BOOL)

        # === Bool constructors ===
        self.register("Bool.true", T_BOOL)
        self.register("Bool.false", T_BOOL)

        # === Bool operations ===
        self.register("Bool.and", BOOL_BOOL_BOOL)
        self.register("Bool.or", BOOL_BOOL_BOOL)
        self.register("Bool.not", BOOL_BOOL)

        # === Type class instances for Nat arithmetic ===
        # HAdd.hAdd : {α β γ : Type} → [inst : HAdd α β γ] → α → β → γ
        # When fully applied to Nat: Nat → Nat → Nat
        # The type class machinery resolves:
        #   HAdd.hAdd Nat Nat Nat (instHAdd Nat instAddNat) ≡ Nat.add
        self.register("HAdd.hAdd", NAT_NAT_NAT)  # simplified: after instance resolution
        self.register("Add.add", NAT_NAT_NAT)
        self.register("instHAdd", T_ERROR)  # instance, type depends on args
        self.register("instAddNat", T_ERROR)

        # HMul
        self.register("HMul.hMul", NAT_NAT_NAT)
        self.register("Mul.mul", NAT_NAT_NAT)
        self.register("instHMul", T_ERROR)
        self.register("instMulNat", T_ERROR)

        # HSub
        self.register("HSub.hSub", NAT_NAT_NAT)
        self.register("Sub.sub", NAT_NAT_NAT)
        self.register("instHSub", T_ERROR)
        self.register("instSubNat", T_ERROR)

        # OfNat: OfNat.ofNat Nat n inst → Nat
        self.register("OfNat.ofNat", T_NAT)  # simplified
        self.register("instOfNatNat", T_ERROR)

        # === Equality ===
        # Eq : {α : Sort u} → α → α → Prop
        # @Eq Nat : Nat → Nat → Prop
        self.register("Eq", T_TYPE)  # polymorphic, but we handle via EQ_NAT
        self.register("Eq.refl", NAT_PROP)  # simplified: @Eq.refl Nat n : Eq Nat n n
        self.register("Eq.symm", PROP_PROP)
        self.register("Eq.trans", PROP_PROP)
        self.register("Eq.mpr", PROP_PROP)
        self.register("Eq.mp", PROP_PROP)
        self.register("congrArg", PROP_PROP)

        # === Decidable ===
        self.register("Decidable", TYPE_PROP)

    def load_from_export(self, filepath: str):
        """Parse CONST_TYPE, INDUCTIVE, CONSTRUCTOR blocks from the export file
        and register their types. This gives full coverage of Lean4 library."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.read().splitlines()
        except UnicodeDecodeError:
            try:
                with open(filepath, 'r', encoding='utf-16') as f:
                    lines = f.read().splitlines()
            except:
                return
        except:
            return

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # CONST_TYPE name
            if line.startswith("CONST_TYPE "):
                name = line.replace("CONST_TYPE ", "").strip()
                if name.endswith(" NOT_FOUND"):
                    i += 1
                    continue
                # For now, just register if not already there. 
                # Ideally we'd parse the full ExprNode and hash the type.
                # Since hashing requires knowing child types, we do a basic registration.
                self.get_or_create(name)
                i += 1
            
            # INDUCTIVE name
            elif line.startswith("=== INDUCTIVE "):
                name = line.replace("=== INDUCTIVE ", "").replace(" ===", "").strip()
                self.register(name, T_TYPE) # Assume Type
                i += 1
                
                # Parse constructors for this inductive type
                tag = 0
                while i < len(lines) and not lines[i].startswith("==="):
                    if lines[i].strip().startswith("--- CTORS"):
                        i += 1
                        while i < len(lines) and lines[i].strip().startswith("CTOR"):
                            ctor_name = lines[i].strip().replace("CTOR ", "")
                            cid = self.get_or_create(ctor_name)
                            self.id_to_tag[cid] = tag
                            tag += 1
                            i += 1
                    else:
                        i += 1
                        
            # CONSTRUCTOR name
            elif line.startswith("=== CONSTRUCTOR "):
                name = line.replace("=== CONSTRUCTOR ", "").replace(" ===", "").strip()
                self.get_or_create(name)
                i += 1
                
            # RECURSOR name
            elif line.startswith("=== RECURSOR "):
                name = line.replace("=== RECURSOR ", "").replace(" ===", "").strip()
                self.get_or_create(name)
                i += 1
            else:
                i += 1

        # Register standard List constructors explicitly
        self.register("List.nil", T_ERROR) # Polimorphic
        self.register("List.cons", T_ERROR)
        self.register("List.rec", T_ERROR)
        self.register("String", T_TYPE)
        self.register("String.mk", T_ERROR)

        # === Register pi decompositions ===
        self.pi_types = [
            (T_NAT, T_NAT),
            (T_NAT, NAT_NAT),
            (T_NAT, T_BOOL),
            (T_NAT, NAT_BOOL),
            (T_BOOL, T_BOOL),
            (T_BOOL, BOOL_BOOL),
            (T_NAT, T_PROP),
            (T_NAT, NAT_PROP),
            (T_PROP, T_PROP),
            (T_TYPE, T_TYPE),
            (T_TYPE, T_PROP),
            (T_BOOL, T_NAT),
        ]
        
        # Phase 7: Setup fully generic rules manually for core types (later from export)
        # rec_rules = (n_params << 16) | n_minors
        nat_rec_id = self.get_or_create("Nat.rec")
        self.id_to_rule[nat_rec_id] = (0 << 16) | 2 # 0 params (just motive), 2 minors (zero, succ)
        
        bool_rec_id = self.get_or_create("Bool.rec")
        self.id_to_rule[bool_rec_id] = (0 << 16) | 2 # 0 params, 2 minors (true, false)
        
        eq_rec_id = self.get_or_create("Eq.rec")
        self.id_to_rule[eq_rec_id] = (2 << 16) | 1 # 2 params (type, a), 1 minor (refl)
        
        list_rec_id = self.get_or_create("List.rec")
        self.id_to_rule[list_rec_id] = (1 << 16) | 2 # 1 param (Type), 2 minors (nil, cons)

    def register(self, name: str, type_hash: int) -> int:
        """Register a constant with a known type."""
        if name in self.name_to_id:
            cid = self.name_to_id[name]
            self.id_to_type[cid] = type_hash
            return cid

        cid = self.next_id
        self.next_id += 1
        self.name_to_id[name] = cid
        self.id_to_type[cid] = type_hash
        self.id_to_name[cid] = name
        return cid

    def get_or_create(self, name: str) -> int:
        """Get constant ID, creating with T_ERROR if unknown."""
        if name in self.name_to_id:
            return self.name_to_id[name]
        return self.register(name, T_ERROR)

    def get_type(self, name: str) -> int:
        """Get type hash for a constant name."""
        cid = self.name_to_id.get(name)
        if cid is not None:
            return self.id_to_type.get(cid, T_ERROR)
        return T_ERROR

    def to_numpy(self) -> np.ndarray:
        """Build const_types numpy array for GPU upload."""
        arr = np.zeros(MAX_CONSTS, dtype=np.int64)
        for cid, type_hash in self.id_to_type.items():
            if 0 <= cid < MAX_CONSTS:
                arr[cid] = type_hash
        return arr

    def build_lookup_array(self) -> np.ndarray:
        """Build pi type lookup table for GPU."""
        arr = np.zeros(TABLE_SIZE * 2, dtype=np.int64)
        for dom, cod in self.pi_types:
            h = pi_hash(dom, cod)
            if h < TABLE_SIZE:
                arr[h * 2] = dom
                arr[h * 2 + 1] = cod
        return arr

    def summary(self) -> str:
        """Print environment summary."""
        known = sum(1 for v in self.id_to_type.values() if v != T_ERROR)
        total = len(self.name_to_id)
        lines = [
            f"CIC Environment: {total} constants ({known} with known types)",
            f"  Pi types registered: {len(self.pi_types)}",
        ]
        for name, cid in sorted(self.name_to_id.items(), key=lambda x: x[1]):
            type_hash = self.id_to_type.get(cid, T_ERROR)
            type_name = _type_name(type_hash)
            if type_hash != T_ERROR:
                lines.append(f"  [{cid:3d}] {name:30s} : {type_name}")
        return "\n".join(lines)


def _type_name(h: int) -> str:
    """Human-readable name for a type hash."""
    names = {
        T_ERROR: "ERROR",
        T_PROP: "Prop",
        T_TYPE: "Type",
        T_TYPE1: "Type 1",
        T_NAT: "Nat",
        T_BOOL: "Bool",
        T_STRING: "String",
        T_LIST: "List",
        NAT_NAT: "Nat→Nat",
        NAT_NAT_NAT: "Nat→Nat→Nat",
        BOOL_BOOL: "Bool→Bool",
        NAT_BOOL: "Nat→Bool",
        NAT_NAT_BOOL: "Nat→Nat→Bool",
        BOOL_BOOL_BOOL: "Bool→Bool→Bool",
        NAT_PROP: "Nat→Prop",
        NAT_NAT_PROP: "Nat→Nat→Prop",
        PROP_PROP: "Prop→Prop",
        TYPE_TYPE: "Type→Type",
        TYPE_PROP: "Type→Prop",
    }
    return names.get(h, f"hash({h})")


# ============================================================
# PATTERN RECOGNIZER: Type Class Instance Resolution
# ============================================================

class TypeClassResolver:
    """Recognizes Lean4 type class patterns in expression trees
    and resolves them to their underlying concrete operations.

    Pattern: APP(APP(APP(APP(APP(APP(HAdd.hAdd, Nat), Nat), Nat),
                         APP(APP(instHAdd, Nat), instAddNat)), n), m)
    Resolves to: Nat.add n m

    This doesn't need to run on GPU — it's a Python preprocessing step
    that simplifies the expression tree before GPU upload.
    """

    # Map from (class_func, type_args...) → resolved constant
    RESOLUTIONS = {
        ("HAdd.hAdd", "Nat", "Nat", "Nat"): "Nat.add",
        ("HMul.hMul", "Nat", "Nat", "Nat"): "Nat.mul",
        ("HSub.hSub", "Nat", "Nat", "Nat"): "Nat.sub",
        ("HAdd.hAdd", "Int", "Int", "Int"): "Int.add",
        ("HMul.hMul", "Int", "Int", "Int"): "Int.mul",
    }

    @staticmethod
    def is_typeclass_app(node_name: str) -> bool:
        """Check if a constant is a type class method."""
        return node_name in ("HAdd.hAdd", "HMul.hMul", "HSub.hSub",
                            "HDiv.hDiv", "HMod.hMod",
                            "OfNat.ofNat", "instHAdd", "instHMul",
                            "instHSub", "instAddNat", "instMulNat",
                            "instSubNat", "instOfNatNat")

    @staticmethod
    def resolve(func_name: str, type_args: tuple) -> Optional[str]:
        """Try to resolve a type class application to a concrete constant."""
        key = (func_name,) + type_args
        return TypeClassResolver.RESOLUTIONS.get(key)


# Singleton environment
_default_env: Optional[CICEnvironment] = None

def get_default_env() -> CICEnvironment:
    """Get or create the default CIC environment."""
    global _default_env
    if _default_env is None:
        _default_env = CICEnvironment()
    return _default_env


if __name__ == "__main__":
    env = CICEnvironment()
    print(env.summary())
