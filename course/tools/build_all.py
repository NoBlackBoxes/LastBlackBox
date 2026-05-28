# -*- coding: utf-8 -*-
"""
Rebuild every course README (via the engine) and every enabled docfx site.

Usage:
    python course/tools/build_all.py

Stages:
    0. Regenerate interactive layout SVGs from the static layout_*.svg.
    1. Run the LBB engine, regenerating all `course/versions/<slug>/<NN>_*/README.md`.
    2. For each slug in `SITES`, run `course/tools/docfx/build_site.py <slug>`
       to produce a self-contained docfx site at `docs/<slug>/`.
    3. Write a master sitemap at `docs/sitemap.xml` referencing every
       per-course sitemap.

`SITES` is the explicit list of course slugs whose docfx sites should be
built. Add slugs here as new courses are ready to ship. The engine always
builds READMEs for every entry in `LBB.config.course_names`; only courses
in `SITES` get a docfx site.

Adding a new course slug is zero-config: drop a session tree under
`course/versions/<slug>/` and add the slug to `SITES`. The build picks up
sensible defaults (docfx.json, toc.yml, sessions.toc.yml, sessions.index.md,
template/) from `course/tools/docfx/template/`. Override any of those by
placing a same-named file in `course/versions/<slug>/`.

This is the single command CI/CD should invoke on push to master.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

_HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = _HERE.parents[1]

# Course slugs that have a docfx site to build. Add to this list as more
# courses get `index.md` + `toc.yml` + `docfx.json` authored.
SITES = ["crick"]

# Public base URL the deployed sites will be served from. Used to build the
# top-level master sitemap. Update if Pages config changes.
SITE_BASE_URL = "https://noblackboxes.github.io/LastBlackBox"


def _run(cmd: list[str]) -> None:
    print(f">>> {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(f"command failed (exit {result.returncode}): {' '.join(cmd)}")


def _write_master_sitemap(docs_root: pathlib.Path, sites: list[str], base_url: str) -> None:
    """Write a sitemap-index at docs/sitemap.xml referencing every per-course
    sitemap that exists. Run after the per-course docfx builds.
    """
    entries = []
    for slug in sites:
        per_course = docs_root / slug / "sitemap.xml"
        if per_course.is_file():
            entries.append(f"  <sitemap>\n    <loc>{base_url}/{slug}/sitemap.xml</loc>\n  </sitemap>")
        else:
            print(f"  (skipping {slug}: no sitemap.xml at {per_course.relative_to(docs_root.parent)})")
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        + "\n".join(entries) + "\n"
        '</sitemapindex>\n'
    )
    out = docs_root / "sitemap.xml"
    out.write_text(xml, encoding="utf8")
    print(f"  wrote {out.relative_to(docs_root.parent)} ({len(entries)} sub-sitemap(s))")


def main() -> None:
    python = sys.executable

    # Stage 0: regenerate interactive layout SVGs from the static ones.
    print("=" * 70)
    print("Stage 0: interactive layout SVGs")
    print("=" * 70)
    _run([python, str(REPO_ROOT / "course" / "tools" / "designs" / "generate_interactive_layouts.py")])

    # Stage 1: regenerate all course READMEs via the engine.
    print()
    print("=" * 70)
    print("Stage 1: LBB engine (regenerating course READMEs)")
    print("=" * 70)
    _run([python, str(REPO_ROOT / "course" / "tools" / "engine" / "generate_courses.py")])

    # Stage 2: build a docfx site for each enabled course slug.
    print()
    print("=" * 70)
    print(f"Stage 2: docfx sites for {SITES}")
    print("=" * 70)
    build_site = REPO_ROOT / "course" / "tools" / "docfx" / "build_site.py"
    for slug in SITES:
        _run([python, str(build_site), slug])

    # Stage 3: write the master sitemap referencing each per-course sitemap.
    print()
    print("=" * 70)
    print("Stage 3: master sitemap")
    print("=" * 70)
    _write_master_sitemap(REPO_ROOT / "docs", SITES, SITE_BASE_URL)

    print()
    print("All sites built. Output under:", REPO_ROOT / "docs")


if __name__ == "__main__":
    main()
