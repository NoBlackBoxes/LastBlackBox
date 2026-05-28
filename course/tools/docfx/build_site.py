# -*- coding: utf-8 -*-
"""
Build a docfx site for a single course slug.

Usage:
    python course/tools/docfx/build_site.py <slug>

For example, `python course/tools/docfx/build_site.py crick` produces a
self-contained static site at `docs/<slug>/` from the engine-rendered
READMEs at `course/versions/<slug>/`.

The build is a three-step pipeline:

1. Stage. Wipe `course/versions/<slug>/.docfx-staging/` and assemble the
   docfx layout from a layered set of sources:

     - Shared defaults from `course/tools/docfx/template/`:
         docfx.json (rendered from docfx.json.tmpl), toc.yml,
         template/public/{main.css, main.js}.
     - Per-course overrides from `course/versions/<slug>/`:
         docfx.json, toc.yml, index.md, sessions.toc.yml,
         sessions.index.md, and any files under template/.
     - Auto-generated when missing:
         sessions.toc.yml, sessions.index.md, index.md
         (derived from session template H1s when absent).

2. Bundle assets and rewrite links. Walk every markdown file in staging:
   - Image references (../../../../boxes/<box>/.../*.gif, /boxes/.../*.png,
     etc.) get the referenced file copied into staging/_assets/<box>/...
     and the markdown reference rewritten to a local relative path. The
     site works offline.
   - Other repo-relative refs (datasheets, code folders, sub-projects)
     are rewritten to absolute GitHub URLs so they still resolve.

3. Run docfx build, writing the static site to `docs/<slug>/`.

The script is course-agnostic: any course slug whose folder has at least
one NN_*/README.md can be built. Customise per-course by dropping
docfx.json / toc.yml / index.md / sessions.* / template/ files into
`course/versions/<slug>/`.
"""
from __future__ import annotations

import pathlib
import re
import shutil
import subprocess
import sys

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from rewrite_links import IMG_EXT, rewrite_non_images, _PARENT_TRAVERSAL  # noqa: E402

REPO_ROOT = _HERE.parents[2]
STAGING_NAME = ".docfx-staging"

# Shared brand assets used by every course site. Copied into staging root so
# docfx's _appLogoPath / _appFaviconPath can find them.
BRAND_LOGO    = REPO_ROOT / "course" / "_resources" / "designs" / "logo" / "icons" / "icon_LBB.svg"
BRAND_FAVICON = REPO_ROOT / "course" / "_resources" / "designs" / "logo" / "icons" / "icon_LBB_256.png"

# Shared docfx defaults (template + CSS/JS shared across all courses).
SHARED_TEMPLATE_DIR = _HERE / "template"

# Public base URL for sitemap generation. Mirrors build_all.SITE_BASE_URL.
# Kept here too so build_site.py can be invoked standalone.
SITE_BASE_URL = "https://noblackboxes.github.io/LastBlackBox"


# Match relative parent-traversal AND absolute repo-root image references.
# Captures the path text we need to resolve to an absolute repo path.
# `data=` is included so <object data="..."> embeds (interactive SVGs) bundle too.
_ROOTS = r"(?:boxes|libs|course)"
_HTML_IMG_RE = re.compile(
    rf'((?:href|src|data)\s*=\s*")((?:/|(?:\.\./)+){_ROOTS}/[^"]*?\.(?:'
    + "|".join(sorted(IMG_EXT)) + r'))(")',
    re.IGNORECASE,
)
_MD_IMG_RE = re.compile(
    rf'(\]\()((?:/|(?:\.\./)+){_ROOTS}/[^)\s]+?\.(?:'
    + "|".join(sorted(IMG_EXT)) + r'))(\))',
    re.IGNORECASE,
)

# First H1 in a session template.md is `# CourseName : SessionName`. Capture
# everything after the colon as the human-readable session label.
_H1_SESSION_RE = re.compile(r"^#\s+[^:]+?\s+:\s+(.+?)\s*$", re.MULTILINE)


def _normalise_to_repo_relative(path: str) -> str:
    """Strip leading parent-traversal or root slash, return e.g. 'boxes/.../X.gif'."""
    return _PARENT_TRAVERSAL.sub("", path).lstrip("/")


def _bundle_assets_in_markdown(
    md_path: pathlib.Path, staging_root: pathlib.Path, repo_root: pathlib.Path
) -> tuple[str, list[str]]:
    """Read *md_path*, bundle every image ref into staging/_assets/, return
    (rewritten_text, list_of_bundled_repo_paths).
    """
    text = md_path.read_text(encoding="utf8")
    bundled: list[str] = []

    # Relative prefix from this markdown file to staging_root/_assets/.
    depth = len(md_path.relative_to(staging_root).parts) - 1
    prefix = ("../" * depth) + "_assets/"

    def _handle(repo_rel: str) -> str | None:
        src = repo_root / repo_rel
        if not src.is_file():
            return None  # leave the ref alone if the file is missing
        dst = staging_root / "_assets" / repo_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)
        bundled.append(repo_rel)
        return prefix + repo_rel

    def _html_sub(m):
        head, path, tail = m.group(1), m.group(2), m.group(3)
        repo_rel = _normalise_to_repo_relative(path)
        new = _handle(repo_rel)
        if new is None:
            return m.group(0)
        return f"{head}{new}{tail}"

    def _md_sub(m):
        head, path, tail = m.group(1), m.group(2), m.group(3)
        repo_rel = _normalise_to_repo_relative(path)
        new = _handle(repo_rel)
        if new is None:
            return m.group(0)
        return f"{head}{new}{tail}"

    text = _HTML_IMG_RE.sub(_html_sub, text)
    text = _MD_IMG_RE.sub(_md_sub, text)
    return text, bundled


def _humanise_dirname(name: str) -> str:
    """Fallback session label when no template.md H1 is found.
    `01_electronics-and-sensors` -> `Electronics and Sensors`.
    """
    stem = re.sub(r"^[0-9]+_", "", name).replace("-", " ").replace("_", " ")
    return stem.strip().title()


def _session_label(session_dir: pathlib.Path) -> str:
    """Return the human-readable label for a session. Prefer the first H1
    in `_resources/template.md` (format `# CourseName : SessionName`); fall
    back to humanised directory name.
    """
    tpl = session_dir / "_resources" / "template.md"
    if tpl.is_file():
        text = tpl.read_text(encoding="utf8", errors="ignore")
        m = _H1_SESSION_RE.search(text)
        if m:
            return m.group(1).strip()
    return _humanise_dirname(session_dir.name)


def _discover_sessions(course_dir: pathlib.Path) -> list[tuple[pathlib.Path, str]]:
    """Return [(session_dir, label), ...] sorted by directory name."""
    out = []
    for session_dir in sorted(course_dir.glob("[0-9][0-9]_*")):
        if not session_dir.is_dir():
            continue
        if not (session_dir / "README.md").is_file():
            continue
        out.append((session_dir, _session_label(session_dir)))
    return out


def _generate_sessions_toc(sessions: list[tuple[pathlib.Path, str]]) -> str:
    """Default sessions/toc.yml content from a list of (dir, label) pairs."""
    lines = [
        "# Auto-generated by build_site.py from session template.md H1s.",
        "# Override by placing a hand-authored sessions.toc.yml next to docfx.json.",
        "",
        "- name: Overview",
        "  href: index.md",
        "",
    ]
    for session_dir, label in sessions:
        lines.append(f"- name: {label}")
        lines.append(f"  href: {session_dir.name}/README.md")
        lines.append("")
    return "\n".join(lines)


def _generate_sessions_index(slug: str, sessions: list[tuple[pathlib.Path, str]]) -> str:
    """Default sessions/index.md content."""
    lines = [f"# Sessions", ""]
    lines.append("Pick a session to begin. Work through them in order.")
    lines.append("")
    for session_dir, label in sessions:
        lines.append(f"- [{label}]({session_dir.name}/README.md)")
    lines.append("")
    return "\n".join(lines)


def _generate_index(slug: str, app_name: str, sessions: list[tuple[pathlib.Path, str]]) -> str:
    """Default landing page when a course has no hand-written index.md."""
    lines = [
        f"# {app_name}",
        "",
        f"A LastBlackBox course version (`{slug}`). Pick a session to begin.",
        "",
        "## Sessions",
        "",
    ]
    for session_dir, label in sessions:
        lines.append(f"- [{label}](sessions/{session_dir.name}/README.md)")
    lines.append("")
    return "\n".join(lines)


def _render_docfx_json(slug: str, app_name: str) -> str:
    """Render docfx.json from the shared template using simple format-string
    substitution. JSON braces in the template are doubled to escape them.
    """
    tmpl = (SHARED_TEMPLATE_DIR / "docfx.json.tmpl").read_text(encoding="utf8")
    return tmpl.format(slug=slug, app_name=app_name, site_base_url=SITE_BASE_URL)


def _copy_tree_overlay(src: pathlib.Path, dst: pathlib.Path) -> None:
    """Copy every file in *src* into *dst*, creating directories as needed.
    Overwrites existing files (so later overlays beat earlier ones).
    """
    if not src.is_dir():
        return
    for path in src.rglob("*"):
        if path.is_dir():
            continue
        rel = path.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _default_app_name(slug: str) -> str:
    """Default app name for a course slug when no docfx.json override exists.
    `crick` -> `LBB: Crick`. Hyphens become spaces; words title-cased.
    """
    pretty = slug.replace("-", " ").replace("_", " ").title()
    return f"LBB: {pretty}"


def _stage(course_dir: pathlib.Path, staging: pathlib.Path, slug: str) -> list[pathlib.Path]:
    """Wipe and recreate *staging*; assemble docfx layout from shared defaults
    + per-course overrides + auto-generated fallbacks. Returns list of
    markdown files placed in staging that need link rewriting.
    """
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    sessions = _discover_sessions(course_dir)

    # ---- docfx.json: per-course override beats rendered default --------------
    course_docfx = course_dir / "docfx.json"
    if course_docfx.is_file():
        shutil.copy2(course_docfx, staging / "docfx.json")
        app_name = _default_app_name(slug)  # only used by generated fallbacks
    else:
        app_name = _default_app_name(slug)
        (staging / "docfx.json").write_text(_render_docfx_json(slug, app_name), encoding="utf8")

    # ---- toc.yml: per-course override beats shared default -------------------
    course_toc = course_dir / "toc.yml"
    if course_toc.is_file():
        shutil.copy2(course_toc, staging / "toc.yml")
    else:
        shutil.copy2(SHARED_TEMPLATE_DIR / "toc.yml", staging / "toc.yml")

    # ---- template/: shared CSS/JS first, per-course overrides on top --------
    _copy_tree_overlay(SHARED_TEMPLATE_DIR / "template", staging / "template")
    _copy_tree_overlay(course_dir / "template",         staging / "template")

    # ---- Brand assets (logo + favicon) at staging root ----------------------
    if BRAND_LOGO.is_file():
        shutil.copy2(BRAND_LOGO, staging / "logo.svg")
    if BRAND_FAVICON.is_file():
        shutil.copy2(BRAND_FAVICON, staging / "favicon.png")

    # ---- index.md: per-course override beats auto-generated fallback --------
    course_index = course_dir / "index.md"
    if course_index.is_file():
        shutil.copy2(course_index, staging / "index.md")
    else:
        (staging / "index.md").write_text(
            _generate_index(slug, app_name, sessions), encoding="utf8"
        )

    md_files: list[pathlib.Path] = [staging / "index.md"]

    # ---- sessions/ subtree: docfx wants sessions/toc.yml + sessions/<NN>/README.md
    sessions_root = staging / "sessions"
    sessions_root.mkdir(parents=True, exist_ok=True)

    course_sessions_toc = course_dir / "sessions.toc.yml"
    if course_sessions_toc.is_file():
        shutil.copy2(course_sessions_toc, sessions_root / "toc.yml")
    else:
        (sessions_root / "toc.yml").write_text(
            _generate_sessions_toc(sessions), encoding="utf8"
        )

    course_sessions_index = course_dir / "sessions.index.md"
    if course_sessions_index.is_file():
        shutil.copy2(course_sessions_index, sessions_root / "index.md")
    else:
        (sessions_root / "index.md").write_text(
            _generate_sessions_index(slug, sessions), encoding="utf8"
        )
    md_files.append(sessions_root / "index.md")

    for session_dir, _label in sessions:
        dst_dir = sessions_root / session_dir.name
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / "README.md"
        shutil.copy2(session_dir / "README.md", dst)
        md_files.append(dst)

    return md_files


def _bundle_and_rewrite(md_files: list[pathlib.Path], staging: pathlib.Path) -> int:
    """Bundle assets and rewrite links across every markdown in staging.
    Returns the number of unique assets bundled.
    """
    all_bundled: set[str] = set()
    for md in md_files:
        text, bundled = _bundle_assets_in_markdown(md, staging, REPO_ROOT)
        text = rewrite_non_images(text)
        md.write_text(text, encoding="utf8")
        all_bundled.update(bundled)
    return len(all_bundled)


def _run_docfx(staging: pathlib.Path, output: pathlib.Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    cmd = ["docfx", "build", str(staging / "docfx.json"), "--output", str(output)]
    print(f"  running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(f"docfx build failed (exit {result.returncode})")


def _inject_custom_js(output: pathlib.Path) -> int:
    """docfx's modern template auto-links public/main.css but NOT public/main.js.
    Walk every rendered .html in *output*, inject a <script> tag for our custom
    main.js before </body>. Returns number of files patched.
    """
    if not (output / "public" / "main.js").is_file():
        return 0
    tag_marker = 'data-lbb-injected="1"'
    patched = 0
    for html_path in output.rglob("*.html"):
        try:
            text = html_path.read_text(encoding="utf8")
        except UnicodeDecodeError:
            continue
        if tag_marker in text or "</body>" not in text:
            continue
        # Compute relative path from this html file to <output>/public/main.js.
        depth = len(html_path.relative_to(output).parts) - 1
        prefix = "../" * depth
        tag = f'<script {tag_marker} defer src="{prefix}public/main.js"></script>'
        text = text.replace("</body>", tag + "\n</body>", 1)
        html_path.write_text(text, encoding="utf8")
        patched += 1
    return patched


def build(slug: str) -> None:
    course_dir = REPO_ROOT / "course" / "versions" / slug
    staging    = course_dir / STAGING_NAME
    output     = REPO_ROOT / "docs" / slug

    if not course_dir.is_dir():
        raise SystemExit(f"course folder not found: {course_dir}")

    print(f"[{slug}] staging at {staging.relative_to(REPO_ROOT)}")
    md_files = _stage(course_dir, staging, slug)
    print(f"[{slug}] copied {len(md_files)} markdown file(s)")

    print(f"[{slug}] bundling assets and rewriting links")
    n_assets = _bundle_and_rewrite(md_files, staging)
    print(f"[{slug}] bundled {n_assets} unique asset(s) into _assets/")

    print(f"[{slug}] building docfx site -> {output.relative_to(REPO_ROOT)}")
    # Wipe the previous output so renamed/removed pages don't linger.
    if output.exists():
        shutil.rmtree(output)
    _run_docfx(staging, output)

    n_patched = _inject_custom_js(output)
    if n_patched:
        print(f"[{slug}] injected custom main.js into {n_patched} html file(s)")
    print(f"[{slug}] done.")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: build_site.py <course-slug>")
    build(sys.argv[1])


if __name__ == "__main__":
    main()
