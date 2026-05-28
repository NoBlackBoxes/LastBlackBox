# -*- coding: utf-8 -*-
"""
Rewrite repo-relative links to absolute GitHub URLs.

Course content lives in `course/`, `boxes/`, and `libs/` and links between
these directories using either absolute repo paths (`/boxes/...`) or relative
parent-traversal paths (`../../../../boxes/...`). When rendered into a docfx
site at a different depth, those paths break.

This module rewrites non-image links to absolute URLs:
    - file extensions      -> github.com/.../blob/<branch>/...
    - folders (no ext)     -> github.com/.../tree/<branch>/...

Image refs are intentionally NOT rewritten here; the build_site.py asset
bundler handles those by copying files into the staging tree and rewriting
to local relative paths.
"""
import re

GH_OWNER  = "NoBlackBoxes"
GH_REPO   = "LastBlackBox"
GH_BRANCH = "master"
GH_BASE   = f"https://github.com/{GH_OWNER}/{GH_REPO}"
RAW_BASE  = f"https://raw.githubusercontent.com/{GH_OWNER}/{GH_REPO}/{GH_BRANCH}"

IMG_EXT = {"png", "jpg", "jpeg", "gif", "svg", "webp", "ico"}

_PARENT_TRAVERSAL = re.compile(r"^(?:\.\./)+")
_ROOTS            = r"(?:boxes|libs|course)"
_HTML_ATTR_RE     = re.compile(rf'(href|src)\s*=\s*"((?:/|(?:\.\./)+){_ROOTS}/[^"]*)"')
_MD_LINK_RE       = re.compile(rf'\]\(((?:/|(?:\.\./)+){_ROOTS}/[^)\s]+)\)')


def _last_seg_ext(path: str) -> str | None:
    cleaned = _PARENT_TRAVERSAL.sub("", path).lstrip("/").rstrip("/")
    if not cleaned:
        return None
    last = cleaned.rsplit("/", 1)[-1]
    if "." not in last:
        return None
    return last.rsplit(".", 1)[-1].lower()


def _is_image_path(path: str) -> bool:
    ext = _last_seg_ext(path)
    return ext in IMG_EXT if ext else False


def _to_github_url(path: str) -> str:
    cleaned = _PARENT_TRAVERSAL.sub("", path).lstrip("/").rstrip("/")
    if not cleaned:
        return f"{GH_BASE}/tree/{GH_BRANCH}"
    ext = _last_seg_ext(path)
    if ext is None:
        return f"{GH_BASE}/tree/{GH_BRANCH}/{cleaned}"
    return f"{GH_BASE}/blob/{GH_BRANCH}/{cleaned}"


def rewrite_non_images(text: str) -> str:
    """Rewrite repo-relative non-image links in *text* to GitHub URLs.

    Image refs (anything ending in a known image extension) are left untouched
    so the asset bundler can pick them up.
    """
    def html_sub(m):
        attr, path = m.group(1), m.group(2)
        if _is_image_path(path):
            return m.group(0)
        return f'{attr}="{_to_github_url(path)}"'

    def md_sub(m):
        path = m.group(1)
        if _is_image_path(path):
            return m.group(0)
        return f"]({_to_github_url(path)})"

    text = _HTML_ATTR_RE.sub(html_sub, text)
    text = _MD_LINK_RE.sub(md_sub, text)
    return text
