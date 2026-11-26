# -*- coding: utf-8 -*-
"""Build the unified Last Black Box Zensical site.

Usage:
    uv run --locked python course/tools/zensical/build_site.py

Call tree:
    module execution
    ├── import LBB.config
    │   └── load and validate `course/courses.json`
    └── main()
        └── build()
            ├── _discover_sessions() for each published course
            ├── _render_site_config()
            │   └── _render_native_theme()
            ├── _render_catalogue_page()
            ├── _stage_course() for each published course
            │   ├── _generate_index() when no authored `index.md` exists
            │   ├── _render_partner()
            │   ├── _write_landing_page()
            │   ├── _render_session_navigation() for each session page
            │   └── _insert_curriculum_map()
            │       └── _render_curriculum_map()
            │           └── _curriculum_sessions()
            ├── _bundle_and_rewrite()
            │   ├── _bundle_assets_in_markdown() for each staged page
            │   └── rewrite_non_images() for each staged page
            └── _run_zensical()
                ├── subprocess.run([zensical, build, ...])
                └── copy the rendered site to `docs/`

Inputs:
    - Published-course metadata from `course/courses.json`
    - Course landing pages, generated session READMEs and site assets
    - Shared Zensical styling and catalogue assets

Intermediate and final outputs:
    - Temporary unified Zensical project under `.site-staging/`
    - Rendered website under `docs/`

The build has three stages:

1. Assemble `course/tools/zensical/.site-staging/` from the central course list,
   authored landing pages, engine-generated session READMEs, shared branding and
   optional course partner assets.
2. Adapt engine-owned website blocks, bundle repository images and rewrite
   repository links. Materials become native collapsible tables and linked
   Vimeo thumbnails become native video embeds.
3. Run Zensical in strict mode under `.site-staging/site/`, then replace
   `docs/` with that rendered site.

Zensical owns navigation, breadcrumbs, search, page outlines, palettes and the
responsive page frame. The small shared stylesheet adds readable article and
video sizing, catalogue cards, the LBB curriculum grid and optional partner
identity. The grid is derived from `{Box:Lesson}` tags in the session templates,
so it has no separately maintained course list.
"""
from __future__ import annotations

import html
import json
import pathlib
import re
import shutil
import subprocess
import sys

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from rewrite_links import GH_BASE, IMG_EXT, rewrite_non_images, _PARENT_TRAVERSAL  # noqa: E402

REPO_ROOT = _HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "libs"))

import LBB.config as Config  # noqa: E402

SITE_STAGING = _HERE / ".site-staging"
CATALOGUE_ASSETS = _HERE / "catalogue"
CURRICULUM_MARKER = "<!-- LBB_CURRICULUM_MAP -->"
PARTNER_DIR_NAME = "site"

# Shared brand assets used by every course site.
BRAND_LOGO    = REPO_ROOT / "course" / "_resources" / "designs" / "logo" / "icons" / "icon_NBB.svg"
BRAND_FAVICON = REPO_ROOT / "course" / "_resources" / "designs" / "logo" / "icons" / "icon_NBB_256.png"
ANIMATED_BRAND_LOGO = (
    REPO_ROOT
    / "course"
    / "_resources"
    / "designs"
    / "logo"
    / "animated"
    / "logo_NBB_animated_hover.svg"
)

# Small shared presentation layer on top of Zensical's native theme.
SHARED_STYLESHEET = _HERE / "assets" / "stylesheets" / "extra.css"

# Match relative parent-traversal AND absolute repo-root image references.
# Captures the path text we need to resolve to an absolute repo path.
# `data=` is included so image assets embedded with <object> are bundled too.
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

# The engine renders each lesson video as a linked heading followed by a GIF
# thumbnail. Replace that exact engine-owned block only in the staged website;
# authored lessons and generated repository READMEs stay GitHub-friendly.
_VIMEO_VIDEO_BLOCK_RE = re.compile(
    r'^#### Watch this video: \[(?P<title>[^\]\n]+)\]\('
    r'https://vimeo\.com/(?:manage/videos/)?(?P<id>[0-9]+)[^\s)]*\)\n'
    r'<p align="center">\n'
    r'<a href="[^"]+" title="Control\+Click to watch in new tab">'
    r'<img [^>\n]+/></a>\n'
    r'</p>',
    re.MULTILINE,
)

# The engine wraps its Markdown materials table in raw HTML for GitHub. A
# Markdown parser cannot render a table inside that HTML block, so replace only
# this exact engine-owned wrapper in the staged website. Generated repository
# READMEs remain unchanged.
_MATERIALS_BLOCK_RE = re.compile(
    r"<details><summary><i>Materials</i></summary><p>\s*\n"
    r"(?P<table>Name\|Description\|.*?\n.*?)\n"
    r"</p></details><hr>",
    re.DOTALL,
)

# First H1 in a session template.md is `# CourseName : SessionName`. Capture
# everything after the colon as the human-readable session label.
_H1_SESSION_RE = re.compile(r"^#\s+[^:]+?\s+:\s+(.+?)\s*$", re.MULTILINE)
_LESSON_TAG_RE = re.compile(r"\{([^:{}]+):[^{}]+\}")

def _page_front_matter(description: str | None = None) -> str:
    """Use the native page outline and path without a duplicate nav sidebar."""
    lines = ["---", "hide:", "  - navigation"]
    if description:
        lines.append(f"description: {json.dumps(description, ensure_ascii=False)}")
    lines.extend(["---", ""])
    return "\n".join(lines) + "\n"


def _normalise_to_repo_relative(path: str) -> str:
    """Strip leading parent-traversal or root slash, return e.g. 'boxes/.../X.gif'."""
    return _PARENT_TRAVERSAL.sub("", path).lstrip("/")


def _embed_vimeo_videos(text: str) -> str:
    """Replace engine-rendered Vimeo thumbnails with native Zensical embeds.

    Keep a normal public Vimeo link in the heading as a playback fallback.
    URLs copied from Vimeo's management interface are normalised by video ID.
    """
    def replace(match: re.Match) -> str:
        title = match.group("title")
        video_id = match.group("id")
        public_url = f"https://vimeo.com/{video_id}"
        player_url = f"https://player.vimeo.com/video/{video_id}"
        return (
            f"#### Watch this video: [{title}]({public_url})\n\n"
            f"![type:video]({player_url})"
        )

    return _VIMEO_VIDEO_BLOCK_RE.sub(replace, text)


def _render_materials_tables(text: str) -> str:
    """Render engine-owned materials blocks as native collapsible content."""
    def replace(match: re.Match) -> str:
        table = match.group("table").strip()
        return (
            '<details markdown="1">\n'
            '<summary>Materials</summary>\n\n'
            f'{table}\n\n'
            '</details>'
        )

    return _MATERIALS_BLOCK_RE.sub(replace, text)


def _bundle_assets_in_markdown(
    md_path: pathlib.Path, staging_root: pathlib.Path, repo_root: pathlib.Path
) -> tuple[str, list[str]]:
    """Read *md_path*, bundle every image ref into staging/_assets/, return
    (rewritten_text, list_of_bundled_repo_paths).
    """
    text = md_path.read_text(encoding="utf8")
    text = _render_materials_tables(text)
    text = _embed_vimeo_videos(text)
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


def _generate_index(course: dict, sessions: list[tuple[pathlib.Path, str]]) -> str:
    """Default landing-page body when a course has no hand-written index.md.

    The H1 is added separately from the manifest so the displayed course name
    has one source of truth.
    """
    lines = [
        "## About this course",
        "",
        course["description"],
        "",
        CURRICULUM_MARKER,
        "",
        "## Sessions",
        "",
    ]
    for session_dir, label in sessions:
        lines.append(f"- [{label}](sessions/{session_dir.name}/)")
    lines.append("")
    return "\n".join(lines)

def _render_session_navigation(
    sessions: list[tuple[pathlib.Path, str]],
    session_index: int,
) -> str:
    """Render website-only previous/next links within one course."""
    if len(sessions) < 2:
        return ""

    links = []
    if session_index > 0:
        previous_dir, previous_label = sessions[session_index - 1]
        links.append(
            '<a class="lbb-session-nav__link lbb-session-nav__link--previous" '
            f'rel="prev" href="../{html.escape(previous_dir.name, quote=True)}/">\n'
            '  <span class="lbb-session-nav__direction">Previous session</span>\n'
            f'  <span>{html.escape(previous_label)}</span>\n'
            '</a>'
        )

    if session_index < len(sessions) - 1:
        next_dir, next_label = sessions[session_index + 1]
        links.append(
            '<a class="lbb-session-nav__link lbb-session-nav__link--next" '
            f'rel="next" href="../{html.escape(next_dir.name, quote=True)}/">\n'
            '  <span class="lbb-session-nav__direction">Next session</span>\n'
            f'  <span>{html.escape(next_label)}</span>\n'
            '</a>'
        )

    return (
        '<nav class="lbb-session-nav" aria-label="Session navigation">\n'
        + "\n".join(links)
        + "\n</nav>\n"
    )

def _curriculum_sessions(
    course_dir: pathlib.Path,
    sessions: list[tuple[pathlib.Path, str]],
) -> dict[str, tuple[str, str]]:
    """Map each referenced box to the first session that introduces it.

    Session templates are the source of truth: a `{Box:Lesson}` tag makes that
    box part of the course. Box names are validated against Config.box_names.
    """
    canonical_boxes = {name.casefold(): name for name in Config.box_names}
    curriculum: dict[str, tuple[str, str]] = {}

    for session_dir, session_label in sessions:
        template = session_dir / "_resources" / "template.md"
        if not template.is_file():
            continue
        for raw_name in _LESSON_TAG_RE.findall(template.read_text(encoding="utf8")):
            key = raw_name.strip().casefold()
            if key not in canonical_boxes:
                raise SystemExit(
                    f"unknown box {raw_name!r} in {template.relative_to(REPO_ROOT)}"
                )
            box_name = canonical_boxes[key]
            curriculum.setdefault(box_name, (session_dir.name, session_label))

    return curriculum


def _render_curriculum_map(
    course_dir: pathlib.Path,
    sessions: list[tuple[pathlib.Path, str]],
) -> str:
    """Render the shared curriculum section as accessible HTML."""
    curriculum = _curriculum_sessions(course_dir, sessions)
    open_count = len(curriculum)
    total_count = len(Config.box_names)

    if open_count == total_count:
        intro = "Select any box to open the session where it is first introduced."
    else:
        intro = (
            "Boxes with an open padlock are included in this course. Select one "
            "to open the session where it is first introduced. "
            f'Explore the <a href="{Config.site_base_url}/full/">full course</a> '
            'or browse the remaining '
            'material on '
            f'<a href="{GH_BASE}">GitHub</a>.'
        )

    lines = [
        '<dl class="lbb-course-facts" aria-label="Course overview">',
        '  <div>',
        f'    <dd>{len(sessions)}</dd>',
        '    <dt>sessions</dt>',
        '  </div>',
        '  <div>',
        f'    <dd>{open_count} / {total_count}</dd>',
        '    <dt>boxes opened</dt>',
        '  </div>',
        '</dl>',
        '',
        '<div class="lbb-curriculum-heading">',
        '  <h2 id="the-28-black-boxes">The 28 black boxes</h2>',
        f'  <p>{intro}</p>',
        '</div>',
        '',
        '<div class="lbb-box-grid" '
        'aria-label="Last Black Box curriculum" markdown>',
        '',
    ]

    for box_name in Config.box_names:
        escaped_name = html.escape(box_name)
        if box_name in curriculum:
            session_dir, session_label = curriculum[box_name]
            href = f"sessions/{session_dir}/"
            aria = html.escape(f"{box_name}: open {session_label}", quote=True)
            lines.append(
                f'- [:lucide-lock-open:{{ .lbb-box-lock }} {escaped_name}]({href})'
                f'{{ .lbb-box aria-label="{aria}" }}'
            )
        else:
            aria = html.escape(f"{box_name}: not included in this course", quote=True)
            lines.append(
                f'- <span class="lbb-box" aria-label="{aria}">'
                f':lucide-lock:{{ .lbb-box-lock }} {escaped_name}</span>'
            )

    lines.extend([
        "",
        "</div>",
    ])
    return "\n".join(lines)


def _insert_curriculum_map(
    index_path: pathlib.Path,
    course_dir: pathlib.Path,
    sessions: list[tuple[pathlib.Path, str]],
) -> None:
    """Replace the single curriculum marker in a staged landing page."""
    text = index_path.read_text(encoding="utf8")
    marker_count = text.count(CURRICULUM_MARKER)
    if marker_count != 1:
        raise SystemExit(
            f"expected one {CURRICULUM_MARKER!r} marker in "
            f"{index_path.relative_to(REPO_ROOT)}, found {marker_count}"
        )
    index_path.write_text(
        text.replace(CURRICULUM_MARKER, _render_curriculum_map(course_dir, sessions)),
        encoding="utf8",
    )


def _render_partner(course_dir: pathlib.Path, course_content: pathlib.Path) -> str:
    """Render optional markup from a course's partner assets.

    A course opts in with `site/partner-logo.png` or `.svg`. Optional link and
    accessible-label metadata come from `site/partner.json`.
    """
    site_dir = course_dir / PARTNER_DIR_NAME
    logo = next(
        (site_dir / f"partner-logo.{ext}" for ext in ("png", "svg")
         if (site_dir / f"partner-logo.{ext}").is_file()),
        None,
    )
    if logo is None:
        return ""

    staged_logo = course_content / "assets" / "images" / logo.name
    staged_logo.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(logo, staged_logo)

    metadata: dict = {}
    metadata_path = site_dir / "partner.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise SystemExit(
                f"invalid partner metadata in "
                f"{metadata_path.relative_to(REPO_ROOT)}: {exc}"
            ) from exc
        if not isinstance(metadata, dict):
            raise SystemExit(
                f"partner metadata must be a JSON object: "
                f"{metadata_path.relative_to(REPO_ROOT)}"
            )

    label = metadata.get("label") or "Host institution"
    url = metadata.get("url")
    if not isinstance(label, str) or (url is not None and not isinstance(url, str)):
        raise SystemExit(
            f"partner metadata values must be strings: "
            f"{metadata_path.relative_to(REPO_ROOT)}"
        )

    escaped_label = html.escape(label, quote=True)
    image = (
        f'<img src="assets/images/{html.escape(logo.name, quote=True)}" '
        f'alt="{escaped_label}" class="lbb-secondary-logo">'
    )
    if url:
        image = (
            f'<a href="{html.escape(url, quote=True)}" target="_blank" '
            f'rel="noopener" class="lbb-secondary-logo-link" '
            f'aria-label="{escaped_label}">{image}</a>'
        )

    return "\n".join([
        '<div class="lbb-partner">',
        '  <span class="lbb-partner-label">In partnership with</span>',
        f"  {image}",
        "</div>",
    ])


def _write_landing_page(
    index_path: pathlib.Path,
    course: dict,
    body: str,
    partner: str,
) -> None:
    """Write a staged landing page with its manifest-owned H1.

    Course index files contain only course-specific prose and sections. The
    manifest supplies the displayed name, preventing titles from drifting
    between the catalogue and individual course sites.
    """
    if re.search(r"^#\s+", body, flags=re.MULTILINE):
        raise SystemExit(
            f"course landing-page body must not contain an H1: "
            f"{index_path.relative_to(REPO_ROOT)}"
        )
    sections = [
        _page_front_matter(course["description"]) + f"# {course['name']}"
    ]
    if partner:
        sections.append(partner)
    sections.append(body.lstrip())
    index_path.write_text("\n\n".join(sections), encoding="utf8")


def _toml_string(value: str) -> str:
    """Return a TOML-compatible quoted string."""
    return json.dumps(value, ensure_ascii=False)


def _render_native_theme() -> str:
    """Return shared native theme, palette and Markdown configuration."""
    return """[project.theme]
language = "en"
logo = "assets/images/logo.svg"
favicon = "assets/images/favicon.png"
features = [
  "content.code.copy",
  "navigation.path",
  "navigation.top",
  "search.highlight",
  "toc.follow",
]

[[project.theme.palette]]
scheme = "default"
toggle.icon = "lucide/sun"
toggle.name = "Switch to dark mode"

[[project.theme.palette]]
scheme = "slate"
toggle.icon = "lucide/moon"
toggle.name = "Switch to light mode"

[project.markdown_extensions]
admonition = {}
attr_list = {}
def_list = {}
footnotes = {}
md_in_html = {}
tables = {}
toc.permalink = true
pymdownx.details = {}
pymdownx.emoji.emoji_generator = "zensical.extensions.emoji.to_svg"
pymdownx.emoji.emoji_index = "zensical.extensions.emoji.twemoji"
pymdownx.highlight.anchor_linenums = true
pymdownx.inlinehilite = {}
pymdownx.magiclink = {}
pymdownx.quotes.callouts = true
pymdownx.superfences = {}
pymdownx.tasklist.custom_checkbox = true

[project.plugins.mkdocs-video]
"""


def _render_site_config(
    courses: list[dict],
    sessions_by_slug: dict[str, list[tuple[pathlib.Path, str]]],
) -> str:
    """Render one navigation tree for the catalogue and every course."""
    course_entries = []
    for course in courses:
        slug = course["slug"]
        nav_sessions = ", ".join(
            f"{{ {_toml_string(label)} = "
            f"{_toml_string(f'{slug}/sessions/{session_dir.name}/index.md')} }}"
            for session_dir, label in sessions_by_slug[slug]
        )
        course_nav = (
            f"[{{ \"Overview\" = {_toml_string(f'{slug}/index.md')} }}, "
            f"{{ \"Sessions\" = [{nav_sessions}] }}]"
        )
        course_entries.append(
            f"{{ {_toml_string(course['name'])} = {course_nav} }}"
        )

    nav = "[" + ", ".join(
        ['{ "Courses" = "index.md" }', *course_entries]
    ) + "]"
    project = f"""# Generated by course/tools/zensical/build_site.py. Do not edit.
[project]
site_name = "Last Black Box"
site_url = {_toml_string(Config.site_base_url + "/")}
site_description = "Hands-on robotics courses from atoms to AI."
copyright = "© NoBlackBoxes"
repo_url = "https://github.com/NoBlackBoxes/LastBlackBox"
repo_name = "NoBlackBoxes/LastBlackBox"
docs_dir = "content"
site_dir = "site"
nav = {nav}
extra_css = ["assets/stylesheets/extra.css"]

"""
    return project + _render_native_theme()


def _stage_course(
    content: pathlib.Path,
    course_dir: pathlib.Path,
    course: dict,
    sessions: list[tuple[pathlib.Path, str]],
) -> list[pathlib.Path]:
    """Stage one course inside the unified project content tree."""
    course_content = content / course["slug"]
    course_content.mkdir(parents=True)

    course_index = course_dir / "index.md"
    if course_index.is_file():
        index_body = course_index.read_text(encoding="utf8")
    else:
        index_body = _generate_index(course, sessions)
    _write_landing_page(
        course_content / "index.md",
        course,
        index_body,
        _render_partner(course_dir, course_content),
    )
    _insert_curriculum_map(course_content / "index.md", course_dir, sessions)

    md_files: list[pathlib.Path] = [course_content / "index.md"]

    # Zensical uses folder index pages for clean URLs and native breadcrumbs.
    sessions_root = course_content / "sessions"
    sessions_root.mkdir(parents=True, exist_ok=True)

    for session_index, (session_dir, _label) in enumerate(sessions):
        dst_dir = sessions_root / session_dir.name
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / "index.md"
        source_text = (session_dir / "README.md").read_text(encoding="utf8")
        session_navigation = _render_session_navigation(sessions, session_index)
        staged_text = _page_front_matter(course["description"]) + source_text
        if session_navigation:
            staged_text += (
                "" if staged_text.endswith("\n") else "\n"
            ) + "\n" + session_navigation
        dst.write_text(staged_text, encoding="utf8")
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


def _run_zensical(staging: pathlib.Path, output: pathlib.Path) -> None:
    """Build inside *staging*, then copy the rendered site to *output*."""
    venv = REPO_ROOT / ".venv"
    local_executables = (
        venv / "bin" / "zensical",          # Linux and macOS
        venv / "Scripts" / "zensical.exe",  # Windows
    )
    executable = next(
        (str(path) for path in local_executables if path.is_file()),
        shutil.which("zensical"),
    )
    if executable is None:
        raise SystemExit(
            "Zensical is not installed in the project environment. Run "
            "'uv sync --locked' from the repository root."
        )
    cmd = [
        executable,
        "build",
        "--config-file",
        str(staging / "zensical.toml"),
        "--clean",
        "--strict",
    ]
    print(f"  running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(f"Zensical build failed (exit {result.returncode})")
    shutil.copytree(staging / "site", output)


def _render_catalogue_page(courses: list[dict]) -> str:
    """Render the manifest-owned course list with Zensical's native cards."""
    lines = [
        _page_front_matter(),
        '<object class="lbb-catalogue-logo" '
        'data="assets/images/logo-animated.svg" type="image/svg+xml" '
        'aria-label="Animated NoBlackBoxes logo"></object>',
        "",
        "# NoBlackBoxes courses",
        "",
        "Hands-on robotics courses that open the stack from atoms to artificial intelligence.",
        "",
        '<div class="grid cards course-cards" markdown>',
        "",
    ]
    for course in courses:
        name = html.escape(course["name"])
        slug = html.escape(course["slug"], quote=True)
        description = html.escape(course["description"])
        logo = course.get("logo")
        lines.append(f"-   **[{name}]({slug}/)**")
        if logo:
            logo_name = html.escape(pathlib.Path(logo).name, quote=True)
            lines.extend(["", f"    ![{name} logo](assets/images/{logo_name})"])
        lines.extend(["", f"    {description}", ""])
    lines.extend(["</div>", ""])
    return "\n".join(lines)


def build() -> None:
    """Stage and build the catalogue and all published courses as one site."""
    courses = Config.published_courses
    staging = SITE_STAGING
    output = REPO_ROOT / "docs"
    if staging.exists():
        shutil.rmtree(staging)
    content = staging / "content"
    stylesheets = content / "assets" / "stylesheets"
    images = content / "assets" / "images"
    stylesheets.mkdir(parents=True)
    images.mkdir(parents=True)

    sessions_by_slug: dict[str, list[tuple[pathlib.Path, str]]] = {}
    for course in courses:
        course_dir = REPO_ROOT / "course" / "versions" / course["slug"]
        if not course_dir.is_dir():
            raise SystemExit(f"course folder not found: {course_dir}")
        sessions_by_slug[course["slug"]] = _discover_sessions(course_dir)

    (staging / "zensical.toml").write_text(
        _render_site_config(courses, sessions_by_slug), encoding="utf8"
    )
    (content / "index.md").write_text(
        _render_catalogue_page(courses), encoding="utf8"
    )
    (content / "robots.txt").write_text(
        "User-agent: *\nAllow: /\n\n"
        f"Sitemap: {Config.site_base_url}/sitemap.xml\n",
        encoding="utf8",
    )
    shutil.copy2(SHARED_STYLESHEET, stylesheets / "extra.css")
    shutil.copy2(BRAND_LOGO, images / "logo.svg")
    shutil.copy2(BRAND_FAVICON, images / "favicon.png")
    shutil.copy2(ANIMATED_BRAND_LOGO, images / "logo-animated.svg")

    for course in courses:
        logo = course.get("logo")
        if not logo:
            continue
        source = CATALOGUE_ASSETS / logo
        if not source.is_file():
            raise SystemExit(f"course catalogue logo not found: {source}")
        shutil.copy2(source, images / source.name)

    md_files = [content / "index.md"]
    for course in courses:
        slug = course["slug"]
        course_dir = REPO_ROOT / "course" / "versions" / slug
        staged = _stage_course(
            content, course_dir, course, sessions_by_slug[slug]
        )
        md_files.extend(staged)
        print(f"[{slug}] staged {len(staged)} markdown file(s)")

    print("[site] bundling assets and rewriting links")
    n_assets = _bundle_and_rewrite(md_files, staging / "content")
    print(f"[site] bundled {n_assets} unique asset(s) into _assets/")

    print("[site] building unified Zensical site -> docs/")
    if output.exists():
        shutil.rmtree(output)
    _run_zensical(staging, output)
    print("[site] done.")


def main() -> None:
    if len(sys.argv) != 1:
        raise SystemExit("usage: build_site.py")
    build()


if __name__ == "__main__":
    main()
