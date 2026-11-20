# -*- coding: utf-8 -*-
"""
Rebuild every course README (via the engine) and every enabled docfx site.

Usage:
    python course/tools/build_all.py

Stages:
    0. Regenerate interactive layout SVGs from the static layout_*.svg.
    1. Run the LBB engine, regenerating all `course/versions/<slug>/<NN>_*/README.md`.
    2. For each published course in `course/courses.json`, run
       `course/tools/docfx/build_site.py <slug>`
       to produce a self-contained docfx site at `docs/<slug>/`.
    3. Write a master sitemap at `docs/sitemap.xml` referencing every
       per-course sitemap.
    4. Copy the static root assets (landing page, logos, robots.txt) from
       `course/tools/docfx/root/` into `docs/`.

The course manifest is the single source of truth for engine generation,
published sites, and the root landing-page tiles. The build picks up sensible
docfx defaults from `course/tools/docfx/template/`; a course can override them
under `course/versions/<slug>/`.

This is the single command CI/CD should invoke to build the website.
"""
from __future__ import annotations

import html
import os
import pathlib
import shutil
import subprocess
import sys

_HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = _HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "libs"))

import LBB.config as Config  # noqa: E402

# The LBB engine is a namespace package under libs/. Inject it into PYTHONPATH
# for any subprocess we spawn so `import LBB.config` works without having to set
# up the environment.
_ENV = {
    **os.environ,
    "PYTHONPATH": str(REPO_ROOT / "libs") + os.pathsep + os.environ.get("PYTHONPATH", ""),
}

# Public base URL the deployed sites will be served from. Used to build the
# top-level master sitemap. Update if Pages config changes.
SITE_BASE_URL = "https://noblackboxes.github.io/LastBlackBox"


def _run(cmd: list[str]) -> None:
    print(f">>> {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False, env=_ENV)
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


def _write_root_landing(
    template_path: pathlib.Path,
    output_path: pathlib.Path,
    courses: list[dict],
) -> None:
    """Render the published course tiles into the root landing-page template."""
    marker = "<!-- COURSE_TILES -->"
    template = template_path.read_text(encoding="utf8")
    if template.count(marker) != 1:
        raise SystemExit(f"expected one {marker!r} marker in {template_path}")

    tiles = []
    for course in courses:
        name = html.escape(course["name"])
        slug = html.escape(course["slug"], quote=True)
        description = html.escape(course["description"])
        logo = course.get("logo")

        icon = ""
        item_class = ' class="no-icon"'
        if logo:
            logo_path = template_path.parent / logo
            if not logo_path.is_file():
                raise SystemExit(f"course logo not found: {logo_path}")
            logo_url = html.escape(logo, quote=True)
            icon = f'    <img class="site-icon" src="{logo_url}" alt="{name} logo">\n'
            item_class = ""

        tiles.append(
            f"  <li{item_class}>\n"
            f"{icon}"
            "    <div class=\"body\">\n"
            f"      <a href=\"{slug}/\">{name}</a>\n"
            f"      <div class=\"blurb\">{description}</div>\n"
            f"      <a class=\"hint\" href=\"{slug}/\">Open the course site &rarr;</a>\n"
            "    </div>\n"
            "  </li>"
        )

    output_path.write_text(template.replace(marker, "\n".join(tiles)), encoding="utf8")
    print(f"  rendered {output_path} ({len(courses)} course tile(s))")


def main() -> None:
    python = sys.executable
    docs_root = REPO_ROOT / "docs"
    published_courses = Config.published_courses
    sites = [course["slug"] for course in published_courses]

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
    # Start clean so removed courses and assets cannot survive into a deployment.
    if docs_root.exists():
        shutil.rmtree(docs_root)
    print()
    print("=" * 70)
    print(f"Stage 2: docfx sites for {sites}")
    print("=" * 70)
    build_site = REPO_ROOT / "course" / "tools" / "docfx" / "build_site.py"
    for slug in sites:
        _run([python, str(build_site), slug])

    # Stage 3: write the master sitemap referencing each per-course sitemap.
    print()
    print("=" * 70)
    print("Stage 3: master sitemap")
    print("=" * 70)
    _write_master_sitemap(docs_root, sites, SITE_BASE_URL)

    # Stage 4: copy static root assets (landing page, logos, robots.txt) into
    # docs/. These are hand-authored under course/tools/docfx/root/ and are
    # served at the site root alongside the per-course subtrees.
    print()
    print("=" * 70)
    print("Stage 4: root assets")
    print("=" * 70)
    root_src = REPO_ROOT / "course" / "tools" / "docfx" / "root"
    docs_root.mkdir(parents=True, exist_ok=True)
    for source in sorted(root_src.iterdir()):
        if source.name == "index.html":
            continue
        destination = docs_root / source.name
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(source, destination)
        print(f"  copied {source.name}")
    _write_root_landing(root_src / "index.html", docs_root / "index.html", published_courses)

    print()
    print("All sites built. Output under:", REPO_ROOT / "docs")


if __name__ == "__main__":
    main()
