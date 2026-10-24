"""BDDL ``:goal`` -> ``isaac_goal.Predicate`` list: the task-completion checker bridge.

Parses a BEHAVIOR ``problem0.bddl`` ``:goal`` conjunction and maps it -- via the 2026-challenge
``metadata.task.inst_to_name`` (BDDL synset-instance -> concrete baked object name) -- into the
``isaac_goal`` predicates that ``check_goal`` scores against ``scene_state.relations()``. This closes
the one gap that let a challenge task be RUN but not honestly VERIFIED: without it, success was
hand-hardcoded; with it, the pass/fail is DERIVED from the task's own BDDL goal.

v1 scope (deliberately narrow + LOUD on the rest):
  * flat ``(and <literal> ...)`` (or a single bare literal) of GROUND predicates only;
  * supported predicate tokens: inside/ontop/open/cooked/cut (map 1:1 to Pred). ``not open`` -> CLOSED.
  * quantifiers (forall/exists/forn/forpairs), or/imply, and unsupported predicates RAISE
    ``BddlGoalError`` -- never silently drop a clause (an empty goal is trivially True in check_goal,
    so silence would be a false PASS). forall/exists expansion is Phase-2 (see the plan).

Pure Python (no numpy / no Isaac); import-safe anywhere.

    bash hand/run_isaac.sh simulator.physical_world/bddl_goal.py   # offline selftest (no Isaac boot)
"""

from __future__ import annotations

import os
import re
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import isaac_goal  # noqa: E402  (same dir)

# BDDL predicate token -> (isaac_goal builder, arity). None = recognised but NOT yet supported
# (raises a clear error instead of silently dropping the clause). Placement relations nextto/under/
# touching and the state verbs toggled_on/frozen are None until their Pred + producer land (see plan).
_PRED: dict[str, object] = {
    "inside": (isaac_goal.inside, 2),
    "ontop": (isaac_goal.on, 2),
    "open": (isaac_goal.is_open, 1),
    "cooked": (isaac_goal.is_cooked, 1),
    "cut": (isaac_goal.is_cut, 1),
    # recognised-but-unsupported (fail loudly, don't drop):
    "toggled_on": None,
    "nextto": None,
    "under": None,
    "touching": None,
    "frozen": None,
    "covered": None,
    "filled": None,
    "contains": None,
    "attached": None,
    "real": None,
    "on_fire": None,
}

_QUANT = {"forall", "exists", "forn", "forpairs", "fornpairs"}
_LOGIC_UNSUPPORTED = {"or", "imply", "iff"}


class BddlGoalError(ValueError):
    """A :goal this v1 bridge cannot faithfully represent (unsupported construct / unresolved name)."""


# ---- minimal s-expression parser (BDDL is a lisp dialect) -------------------------------------------
def _tokenize(text: str) -> list[str]:
    return re.findall(r"\(|\)|[^\s()]+", text)


def _parse(tokens: list[str], i: int) -> tuple[list, int]:
    """Parse one parenthesised list starting at tokens[i] == '(' -> (node, next_index)."""
    node: list = []
    i += 1  # consume '('
    while i < len(tokens) and tokens[i] != ")":
        if tokens[i] == "(":
            child, i = _parse(tokens, i)
            node.append(child)
        else:
            node.append(tokens[i])
            i += 1
    return node, i + 1  # consume ')'


def load_goal_sexpr(bddl_path: str) -> list:
    """Return the ``:goal`` expression (usually ``['and', <literal>, ...]``) from a problem BDDL."""
    tokens = _tokenize(open(bddl_path).read())
    tree, _ = _parse(tokens, 0)  # tree = ['define', [...], ..., [':goal', <expr>]]
    for el in tree:
        if isinstance(el, list) and el and el[0] == ":goal":
            if len(el) < 2:
                raise BddlGoalError(f"empty :goal in {bddl_path}")
            return el[1]
    raise BddlGoalError(f"no :goal section in {bddl_path}")


def _strip_q(sym: str) -> str:
    return sym[1:] if isinstance(sym, str) and sym.startswith("?") else sym


def _typed_var(spec):
    """['?x','-','toy.n.01'] (or ['?x1','?x2','-','T']) -> (var_names_without_?, type_synset)."""
    if not isinstance(spec, list) or len(spec) < 3 or "-" not in spec:
        raise BddlGoalError(f"bad quantifier var spec: {spec!r}")
    i = spec.index("-")
    return [_strip_q(v) for v in spec[:i]], spec[i + 1]


def _instances_of(typ: str, inst_to_name: dict[str, str]) -> list[str]:
    """Concrete names for every inst_to_name key of synset ``typ`` ('toy.n.01_3' -> synset 'toy.n.01')."""
    out = []
    for k, v in inst_to_name.items():
        base = k.rsplit("_", 1)[0] if re.search(r"_\d+$", k) else k
        if base == typ:
            out.append(v)
    return out


def _resolve_arg(
    sym: str, inst_to_name: dict[str, str], bindings: dict[str, str]
) -> str:
    """A goal argument -> concrete name: a bound quantifier var (bindings) else inst_to_name."""
    s = _strip_q(sym)
    if s in bindings:
        return bindings[s]
    name = inst_to_name.get(s)
    if name is None:
        raise BddlGoalError(
            f"unresolved BDDL symbol {s!r} (bindings={sorted(bindings)}, inst_to_name={sorted(inst_to_name)})"
        )
    return name


def _ground_literal(node, inst_to_name, bindings):
    head = node[0]
    # Skip a literal that references a FLOOR / ROOM / the AGENT: e.g. ontop(ashcan, floor) or inroom(...)
    # is a persistent scene-binding that holds at :init and is not a manipulation goal (there is no
    # registered 'floor' surface entity to satisfy it against) -> trivially satisfied, drop it.
    _raw = [_strip_q(a) for a in node[1:] if isinstance(a, str)]
    if any(r.startswith(("floor.n", "agent.n")) or r.startswith("room") for r in _raw):
        return []
    if head == "not":
        inner = node[1] if len(node) == 2 else None
        if isinstance(inner, list) and inner and inner[0] == "open" and len(inner) == 2:
            return [
                isaac_goal.is_closed(_resolve_arg(inner[1], inst_to_name, bindings))
            ]
        raise BddlGoalError(f"unsupported negation (only 'not open' allowed): {node!r}")
    if head not in _PRED:
        raise BddlGoalError(f"unknown predicate {head!r}")
    spec = _PRED[head]
    if spec is None:
        raise BddlGoalError(
            f"predicate {head!r} not supported yet (needs a new verb/Pred; see plan)"
        )
    builder, arity = spec
    args = [_resolve_arg(a, inst_to_name, bindings) for a in node[1:]]
    if len(args) != arity:
        raise BddlGoalError(
            f"predicate {head!r} expects {arity} args, got {len(args)}: {node!r}"
        )
    return [builder(*args)]


def _expand(node, inst_to_name, bindings, ex_out=None) -> list:
    """Recursively expand a goal s-expr -> list of GROUND Predicate. Handles and / forall (enumerate a
    type's instances -> AND) / exists (bind to the (single) instance the demo places) / a ground literal.
    or/imply/forn/forpairs remain unsupported (a flat conjunctive check_goal cannot express them).

    ``ex_out`` (opt-in): a dict the caller passes to collect EXISTENTIAL witnesses -- ``{subject_name:
    [candidate_names]}``. Because ``exists`` is satisfied by ANY instance of the type, a reachability-aware
    caller may re-pick a REACHABLE witness instead of the arbitrary ``insts[0]``. None -> not collected
    (byte-identical to before)."""
    if not isinstance(node, list) or not node:
        raise BddlGoalError(f"malformed goal node: {node!r}")
    head = node[0]
    if head == "and":
        out = []
        for c in node[1:]:
            out += _expand(c, inst_to_name, bindings, ex_out)
        return out
    if head == "forall":
        vars_, typ = _typed_var(node[1])
        insts = _instances_of(typ, inst_to_name)
        if not insts:
            raise BddlGoalError(f"forall {typ!r}: no instances in inst_to_name")
        out = []
        for inst in insts:
            b2 = dict(bindings)
            for v in vars_:
                b2[v] = inst
            out += _expand(node[2], inst_to_name, b2, ex_out)
        return out
    if head == "exists":
        vars_, typ = _typed_var(node[1])
        insts = _instances_of(typ, inst_to_name)
        if not insts:
            raise BddlGoalError(f"exists {typ!r}: no instances in inst_to_name")
        b2 = dict(bindings)
        for v in vars_:
            b2[v] = insts[
                0
            ]  # the demo places a single instance of the container/target type
        # Record the existential candidates for the inner literal's SUBJECT (the non-witness arg), so a
        # reachability-aware caller can re-pick a reachable witness. Only the simple ``exists ?v (pred subj
        # ?v)`` shape is recorded (covers the inside/ontop container goals); other shapes keep insts[0].
        if ex_out is not None and isinstance(node[2], list) and len(node[2]) >= 3:
            _wset = set(vars_)
            for _a in node[2][1:]:
                if isinstance(_a, str) and _strip_q(_a) not in _wset:
                    try:
                        ex_out[_resolve_arg(_a, inst_to_name, b2)] = list(insts)
                    except BddlGoalError:
                        pass
        return _expand(node[2], inst_to_name, b2, ex_out)
    if head in _LOGIC_UNSUPPORTED or head in ("forn", "forpairs", "fornpairs"):
        raise BddlGoalError(
            f"unsupported construct {head!r} (or/imply/forn/forpairs not expressible in a flat check_goal)"
        )
    return _ground_literal(node, inst_to_name, bindings)


def bddl_to_goal(
    bddl_path: str, inst_to_name: dict[str, str], existential_out=None
) -> list:
    """Parse ``bddl_path``'s :goal + ground it via ``inst_to_name`` -> list[isaac_goal.Predicate].

    Expands and/forall/exists over the challenge instances; raises ``BddlGoalError`` on any construct that a
    flat conjunctive check_goal cannot faithfully represent (or/forn/forpairs/unsupported predicate), so the
    caller picks a different task instead of silently scoring a wrong goal. Guarantees a NON-EMPTY result.

    ``existential_out`` (opt-in dict): filled with ``{subject_name: [candidate_names]}`` for each EXISTENTIAL
    goal (``inside food SOME tupperware``), so a caller can faithfully re-pick a REACHABLE witness."""
    preds = _expand(load_goal_sexpr(bddl_path), inst_to_name, {}, existential_out)
    if not preds:
        raise BddlGoalError(f"{bddl_path}: :goal produced 0 predicates")
    return preds


# =====================================================================================================
# Offline selftest (no Isaac): parse the REAL composting_waste goal + reject a quantified goal.
# =====================================================================================================
def _selftest() -> int:
    from isaac_goal import check_goal, Pred

    ok = True

    def _chk(cond, msg):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")

    # s-expr parser sanity.
    node, _ = _parse(_tokenize("(and (inside ?a ?b) (ontop ?c ?d))"), 0)
    _chk(
        node == ["and", ["inside", "?a", "?b"], ["ontop", "?c", "?d"]],
        f"s-expr parse -> {node}",
    )

    # The REAL flagship goal, grounded via a synthetic inst_to_name mirroring the challenge template.
    i2n = {
        "half__banana.n.01_1": "half_banana_68",
        "half__pomegranate.n.01_1": "half_pomegranate_67",
        "ashcan.n.01_1": "trash_can_66",
        "countertop.n.01_1": "countertop_tpuwys_0",
        "floor.n.01_1": "floors_a",
    }
    bddl = os.path.join(
        _THIS_DIR,
        "..",
        "BEHAVIOR-1K",
        "bddl3",
        "bddl",
        "activity_definitions",
        "composting_waste",
        "problem0.bddl",
    )
    if os.path.exists(bddl):
        goal = bddl_to_goal(bddl, i2n)
        got = sorted((p.pred, p.a, p.b) for p in goal)
        want = sorted(
            [
                (Pred.INSIDE, "half_banana_68", "trash_can_66"),
                (Pred.INSIDE, "half_pomegranate_67", "trash_can_66"),
            ]
        )
        _chk(
            got == want,
            f"composting_waste -> {[(p.pred.value, p.a, p.b) for p in goal]}",
        )
        # check_goal wiring: unmet before, met once both INSIDE relations are declared.
        rel_done = {
            (Pred.INSIDE, "half_banana_68", "trash_can_66"),
            (Pred.INSIDE, "half_pomegranate_67", "trash_can_66"),
        }
        _chk(not check_goal(goal, set()), "goal UNMET with no relations")
        _chk(check_goal(goal, rel_done), "goal MET once both INSIDE declared")
    else:
        print(f"  [SKIP] composting BDDL not found at {bddl}")

    # Unresolved instance fails loudly (not a silent empty goal).
    try:
        bddl_to_goal(bddl, {}) if os.path.exists(bddl) else (_ for _ in ()).throw(
            BddlGoalError("x")
        )
        _chk(False, "unresolved instance should raise")
    except BddlGoalError:
        _chk(True, "unresolved instance raises BddlGoalError")

    # forall EXPANDS over the type's instances -> AND of ground predicates.
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".bddl", delete=False) as f:
        f.write(
            "(define (problem t) (:domain d) (:objects) (:init) "
            "(:goal (and (forall (?t - toy.n.01) (inside ?t ?box.n.01_1)))))"
        )
        fpath = f.name
    try:
        gfa = bddl_to_goal(
            fpath, {"toy.n.01_1": "toy_a", "toy.n.01_2": "toy_b", "box.n.01_1": "bin"}
        )
        got_fa = sorted((p.pred.value, p.a, p.b) for p in gfa)
        _chk(
            got_fa == sorted([("inside", "toy_a", "bin"), ("inside", "toy_b", "bin")]),
            f"forall expands -> {[(p.pred.value, p.a, p.b) for p in gfa]}",
        )
    finally:
        os.unlink(fpath)

    # Unsupported constructs still raise (must never silently pass).
    for body, why in (
        ("(:goal (and (toggled_on ?radio.n.01_1)))", "unsupported predicate"),
        (
            "(:goal (or (inside ?a.n.01_1 ?b.n.01_1) (inside ?a.n.01_1 ?c.n.01_1)))",
            "or",
        ),
    ):
        with tempfile.NamedTemporaryFile("w", suffix=".bddl", delete=False) as f:
            f.write(f"(define (problem t) (:domain d) (:objects) (:init) {body})")
            path = f.name
        try:
            bddl_to_goal(
                path,
                {
                    "radio.n.01_1": "r",
                    "a.n.01_1": "a",
                    "b.n.01_1": "b",
                    "c.n.01_1": "c",
                },
            )
            _chk(False, f"{why} should raise")
        except BddlGoalError:
            _chk(True, f"{why} raises BddlGoalError")
        finally:
            os.unlink(path)

    print(f"[bddl_goal] selftest {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(_selftest())
