# -*- coding: utf-8 -*-
"""
Rebuild every course README and the unified Zensical website.

Usage:
    .venv/bin/python course/tools/build_all.py

Stages:
    0. Run the LBB engine, regenerating all `course/versions/<slug>/<NN>_*/README.md`.
    1. Build the root catalogue and all published courses as one Zensical site
       under `docs/`.

The course manifest is the single source of truth for engine generation,
published courses, site metadata, and the root selector cards. Every course
uses Zensical's native theme; course directories provide content and optional
partner assets only.

This is the single command CI/CD should invoke to build the website.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys

_HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = _HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "libs"))

# The LBB engine is a namespace package under libs/. Inject it into PYTHONPATH
# for any subprocess we spawn so `import LBB.config` works without having to set
# up the environment.
_ENV = {
    **os.environ,
    "PYTHONPATH": str(REPO_ROOT / "libs") + os.pathsep + os.environ.get("PYTHONPATH", ""),
}


def _run(cmd: list[str]) -> None:
    print(f">>> {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False, env=_ENV)
    if result.returncode != 0:
        raise SystemExit(f"command failed (exit {result.returncode}): {' '.join(cmd)}")


def main() -> None:
    python = sys.executable

    # Stage 0: regenerate all course READMEs via the engine.
    print("=" * 70)
    print("Stage 0: LBB engine (regenerating course READMEs)")
    print("=" * 70)
    _run([python, str(REPO_ROOT / "course" / "tools" / "engine" / "generate_courses.py")])

    # Stage 1: build the selector and all published courses as one site.
    print()
    print("=" * 70)
    print("Stage 1: unified Zensical website")
    print("=" * 70)
    build_site = REPO_ROOT / "course" / "tools" / "zensical" / "build_site.py"
    _run([python, str(build_site)])

    print()
    print("Website built. Output under:", REPO_ROOT / "docs")


if __name__ == "__main__":
    main()
