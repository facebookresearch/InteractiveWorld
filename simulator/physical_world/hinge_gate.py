#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Regression cover for the DOOR/CONTAINER HINGE WIRING, which offline kinematic checks cannot reach.

WHY THIS FILE EXISTS. Two bugs in ``episode_runtime``'s hinge wiring made episodes die inside
``setup_episode`` with **no manifest and no eval.json**, so they were invisible in every summary built
from manifests:

    1. ``pxr.Tf.ErrorException: The xformOp 'xformOp:translate:hingeTo' already exists``
       -- a glazed cabinet is authored as sibling ``door<N>`` + ``glass<N>`` where only the door
       carries the joint, so the pane is wired once as itself and again riding its frame, and
       ``AddXformOp`` RAISES on a second add.
    2. ``NameError: name 'Gf' is not defined``
       -- a MODULE-LEVEL helper used ``Gf``, but ``episode_runtime`` imports ``pxr`` per-function
       and has no module-level ``Gf``.

**Both pass an offline kinematic check.** Such a check drives a synthetic ``SceneState`` through the
kinematic executor, never builds a USD stage, and cannot even import ``pxr`` (that lives in an Isaac
extscache directory which needs its own entry on ``LD_LIBRARY_PATH``). So no amount of adding
openables to a synthetic scene would cover this code -- a symbolic openable is not a hinge. It needs a
second kind of check, not a bigger scene.

TWO CHECKS, deliberately of different kinds:

  A. STATIC (no pxr, runs anywhere, milliseconds). Every module-level function in ``episode_runtime``
     that uses a ``pxr`` symbol must bind it -- locally or at module scope. This generalises bug (2)
     from an instance to a class, so it also catches later instances.
  B. RUNTIME (needs pxr, ~1 s in a subprocess). ``_wire_hinge_ops`` called TWICE on the same leaf,
     which is bug (1) exactly. Asserts it does not raise, leaves no duplicate op, keeps a stable
     op ORDER, preserves the leaf's pre-existing ops, and lets the SECOND call's pivot win -- which
     is what "the pane rides its frame" requires and is the reason the second call exists at all.

Run standalone::

    python3 simulator/physical_world/hinge_gate.py            # both checks; B self-execs with the right env
    python3 simulator/physical_world/hinge_gate.py --static   # A only, no Isaac paths needed
    python3 simulator/physical_world/hinge_gate.py --runtime  # B only, assumes pxr is already importable

:func:`check` runs both, so a larger gate can include this path. ⚠️ A check that cannot run reports
**SKIP loudly and fails the gate**; silently skipping is the failure mode this whole file exists to
prevent.
"""

from __future__ import annotations

import argparse
import ast
import glob
import os
import subprocess
import sys

_HERE: str = os.path.dirname(os.path.abspath(__file__))
_TARGET: str = os.path.join(_HERE, "episode_runtime.py")

# The pxr submodules this codebase actually uses. A name is only interesting if it is one of these.
_PXR: frozenset[str] = frozenset(
    {
        "Gf",
        "Usd",
        "UsdGeom",
        "UsdPhysics",
        "UsdShade",
        "UsdLux",
        "UsdSkel",
        "Sdf",
        "Tf",
        "Vt",
        "PhysxSchema",
        "UsdUtils",
    }
)


# ===================================================================================================
#  A. static: a module-level function may not use a pxr symbol it never bound
# ===================================================================================================
def _bound_names(node: ast.AST) -> set[str]:
    """pxr submodules brought into scope by an ``import`` anywhere inside ``node``."""
    out: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.ImportFrom) and (n.module or "").split(".")[0] == "pxr":
            out |= {a.asname or a.name for a in n.names}
        elif isinstance(n, ast.Import):
            out |= {(a.asname or a.name).split(".")[0] for a in n.names}
    return out


def _used_names(node: ast.AST) -> set[str]:
    """pxr submodules READ inside ``node``, ignoring nested function bodies (they bind their own)."""
    out: set[str] = set()
    stack = list(ast.iter_child_nodes(node))
    while stack:
        n = stack.pop()
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in _PXR:
            out.add(n.id)
        if isinstance(n, ast.Attribute):
            stack.append(n.value)
            continue
        stack.extend(ast.iter_child_nodes(n))
    return out


def static_check(path: str = _TARGET) -> tuple[bool, list[str]]:
    """Every top-level function that reads a pxr name must have bound it. Returns (ok, lines)."""
    try:
        tree = ast.parse(open(path).read())
    except (OSError, SyntaxError) as exc:
        return False, [f"[hinge-gate] STATIC could not parse {path}: {exc}"]
    module_level = _bound_names_module(tree)
    bad: list[str] = []
    n_fn = 0
    for fn in tree.body:
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        n_fn += 1
        missing = _used_names(fn) - _bound_names(fn) - module_level
        for name in sorted(missing):
            bad.append(
                f"    {os.path.basename(path)}:{fn.lineno} {fn.name}() uses "
                f"pxr.{name} without importing it"
            )
    lines = [
        f"  [{'PASS' if not bad else 'FAIL'}] pxr symbols are bound in all {n_fn} "
        f"module-level functions of {os.path.basename(path)}"
    ]
    return not bad, lines + bad


def _bound_names_module(tree: ast.Module) -> set[str]:
    """Only MODULE-scope imports; a name imported inside some other function does not count here."""
    out: set[str] = set()
    for n in tree.body:
        if isinstance(n, ast.ImportFrom) and (n.module or "").split(".")[0] == "pxr":
            out |= {a.asname or a.name for a in n.names}
        elif isinstance(n, ast.Import):
            out |= {(a.asname or a.name).split(".")[0] for a in n.names}
    return out


# ===================================================================================================
#  B. runtime: _wire_hinge_ops must survive being called twice on the same leaf
# ===================================================================================================
_HINGE_ORDER: tuple[str, ...] = (
    "xformOp:translate:hingeTo",
    "xformOp:rotateZ:doorswing",
    "xformOp:translate:hingeBack",
)


def usd_paths() -> tuple[str, str]:
    """(python path, loader path) for Isaac's bundled USD, found by glob so a version bump survives."""
    root = os.path.join(
        os.path.dirname(os.path.dirname(sys.executable)),
        "lib",
        "python3.11",
        "site-packages",
        "isaacsim",
        "extscache",
    )
    hits = sorted(glob.glob(os.path.join(root, "omni.usd.libs-*")))
    return (hits[-1], os.path.join(hits[-1], "bin")) if hits else ("", "")


def runtime_check() -> tuple[bool, list[str]]:
    """Wire one leaf twice and assert the six properties the two bugs violated."""
    from pxr import (  # noqa: PLC0415 - only importable with the loader path set
        Gf,
        Usd,
        UsdGeom,
    )

    sys.path.insert(0, _HERE)
    import episode_runtime as s3  # noqa: PLC0415

    stage = Usd.Stage.CreateInMemory()
    xf = UsdGeom.Xformable(UsdGeom.Xform.Define(stage, "/door1"))
    xf.AddTranslateOp().Set(Gf.Vec3d(1.0, 2.0, 3.0))
    xf.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    xf.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))
    pre = [o.GetOpName() for o in xf.GetOrderedXformOps()]

    out: list[str] = []
    ok = True

    def chk(cond: bool, msg: str) -> None:
        nonlocal ok
        ok = ok and bool(cond)
        out.append(f"  [{'PASS' if cond else 'FAIL'}] {msg}")

    try:
        s3._wire_hinge_ops(xf, 0.50, 0.25)  # noqa: SLF001
        chk(True, "_wire_hinge_ops: first call on a fresh leaf")
    except Exception as exc:  # noqa: BLE001 - the bug under test is precisely an exception
        chk(False, f"_wire_hinge_ops: first call RAISED {type(exc).__name__}: {exc}")
        return False, out
    first = [o.GetOpName() for o in xf.GetOrderedXformOps()]

    try:
        # THE REGRESSION: a glazed pane is wired as itself, then again riding its frame.
        t_to, _rot, t_back = s3._wire_hinge_ops(xf, -0.75, 0.10)  # noqa: SLF001
        chk(
            True,
            "_wire_hinge_ops: SECOND call on the same leaf (the glazed-pane double wire)",
        )
    except Exception as exc:  # noqa: BLE001
        chk(False, f"_wire_hinge_ops: second call RAISED {type(exc).__name__}: {exc}")
        return False, out

    names = [o.GetOpName() for o in xf.GetOrderedXformOps()]
    chk(
        len(names) == len(set(names)),
        f"no duplicate xformOp after two calls ({len(names)} ops)",
    )
    chk(tuple(names[:3]) == _HINGE_ORDER, "hinge ops lead xformOpOrder, in pivot order")
    chk(names == first, "op order is identical after the second call")
    chk(
        names[3:] == pre, "the leaf's pre-existing ops survive, in their original order"
    )
    chk(
        tuple(t_to.Get()) == (-0.75, 0.10, 0.0)
        and tuple(t_back.Get()) == (0.75, -0.10, 0.0),
        "the SECOND call's pivot wins (the pane rides its frame)",
    )
    return ok, out


# ===================================================================================================
#  driver
# ===================================================================================================
def check() -> tuple[bool, list[str]]:
    """Both checks. B runs in a subprocess so the caller needs no Isaac env itself."""
    ok_a, lines = static_check()
    py, lib = usd_paths()
    if not py:
        return False, lines + [
            "  [SKIP-FAIL] runtime hinge check: Isaac's USD libs were not found. "
            "A skipped check must not read as a pass."
        ]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([py, _HERE, env.get("PYTHONPATH", "")])
    # Isaac's USD needs its own ``bin`` AND the interpreter's ``lib`` (libpython) on the loader path.
    # Both are added explicitly rather than inherited: ``run_isaac.sh`` happens to export the second,
    # but a bare ``python3 hinge_gate.py`` does not, and a check that only runs under one launcher is
    # a check that will quietly stop running.
    own_lib = os.path.join(os.path.dirname(os.path.dirname(sys.executable)), "lib")
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [p for p in (lib, own_lib, env.get("LD_LIBRARY_PATH", "")) if p]
    )
    try:
        p = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--runtime"],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, lines + [
            f"  [SKIP-FAIL] runtime hinge check could not start: {exc}"
        ]
    body = [ln for ln in (p.stdout or "").splitlines() if ln.strip()]
    if p.returncode != 0 and not body:
        body = [
            f"  [FAIL] runtime hinge check exited {p.returncode}: "
            f"{(p.stderr or '').strip().splitlines()[-1][:160] if p.stderr else ''}"
        ]
    return (ok_a and p.returncode == 0), lines + body


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--static", action="store_true")
    ap.add_argument("--runtime", action="store_true")
    args = ap.parse_args(argv)
    if args.static:
        ok, lines = static_check()
    elif args.runtime:
        ok, lines = runtime_check()
    else:
        ok, lines = check()
    print("\n".join(lines))
    print(f"[hinge-gate] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
