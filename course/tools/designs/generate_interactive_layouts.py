# Generate interactive layout SVGs (clickable boxes + hover state)

# Imports
import pathlib
import re

# Specify paths
_HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = _HERE.parents[2]
LAYOUT_DIR = REPO_ROOT / "course" / "_resources" / "designs" / "layout" / "svg"
GH_BASE = "https://github.com/NoBlackBoxes/LastBlackBox/tree/master/boxes"

# Hover CSS injected into every generated SVG. Includes a dark-mode block
# that responds to the OS prefers-color-scheme: dark setting.
STYLE_BLOCK = """
<style type="text/css"><![CDATA[
  a .box-group { cursor: pointer; transition: transform 140ms ease; transform-origin: center; transform-box: fill-box; }
  a:hover .box-group .box   { fill: #ffcc00; stroke: #111; transition: fill 140ms ease, stroke 140ms ease; }
  a:hover .box-group .text  { fill: #111; transition: fill 140ms ease; }
  a:hover .box-group .arrow { fill: #ffcc00; }
  a:hover .box-group        { transform: scale(1.06); }
  a { text-decoration: none; }

  @media (prefers-color-scheme: dark) {
    .box-group .box  { fill: #2a2f36; stroke: #4a525d; }
    .box-group .text { fill: #f0f0f0; }
    .box-group .arrow { fill: #5a626e; }
    a:hover .box-group .box   { fill: #ffcc00; stroke: #ffcc00; }
    a:hover .box-group .text  { fill: #111; }
    a:hover .box-group .arrow { fill: #ffcc00; }
  }
]]></style>
"""


# Wrap each <g id="X"> in an <a> pointing at boxes/x/ on GitHub. Uses both
# SVG 2 `href` and SVG 1.1 `xlink:href` so older renderers also resolve the
# link. `target="_blank"` opens a new tab when the SVG is loaded inline,
# while the onclick fallback escapes the <object> frame for cases where the
# browser would otherwise navigate inside the object.
def wrap_groups(svg):
    count = 0
    onclick = "if(window.top){window.top.open(this.getAttribute('href'),'_blank','noopener');return false;}"
    def replace(match):
        nonlocal count
        name = match.group(1)
        g_match = re.search(r"<g\b[^>]*>", match.group(0))
        if not g_match:
            return match.group(0)
        g_tag = g_match.group(0)
        new_g_tag = re.sub(
            r'<g\s+id="' + re.escape(name) + r'"',
            f'<g id="{name}" class="box-group"',
            g_tag, count=1)
        count += 1
        url = f'{GH_BASE}/{name.lower()}/'
        return (f'<a href="{url}" xlink:href="{url}" target="_blank" rel="noopener" '
                f'onclick="{onclick}" '
                f'aria-label="Open the {name} box on GitHub">'
                f'{new_g_tag}{match.group(2)}</g></a>')
    return re.sub(r'<g\s+id="([A-Za-z0-9_]+)"[^>]*>(.*?)</g>',
                  replace, svg, flags=re.DOTALL), count


# Tighten viewBox + width/height to the actual drawn content (source SVGs use the full LBB canvas)
def tighten_viewbox(svg, margin=0.5):
    xs, ys, rs, bs = [], [], [], []
    for m in re.finditer(r'<rect [^>]*x="([0-9.\-]+)"\s+y="([0-9.\-]+)"\s+width="([0-9.\-]+)"\s+height="([0-9.\-]+)"', svg):
        x, y, w, h = (float(m.group(i)) for i in (1, 2, 3, 4))
        xs.append(x)
        ys.append(y)
        rs.append(x + w)
        bs.append(y + h)
    for m in re.finditer(r'<polygon [^>]*points="([^"]+)"', svg):
        nums = [float(c) for c in re.findall(r'(-?\d+(?:\.\d+)?)', m.group(1))]
        xs.extend(nums[0::2])
        ys.extend(nums[1::2])
        rs.extend(nums[0::2])
        bs.extend(nums[1::2])
    for m in re.finditer(r'<text [^>]*x="([0-9.\-]+)"\s+y="([0-9.\-]+)"', svg):
        x, y = float(m.group(1)), float(m.group(2))
        xs.append(x)
        ys.append(y)
        rs.append(x)
        bs.append(y)
    if not xs:
        return svg
    bx, by = min(xs) - margin, min(ys) - margin
    bw, bh = (max(rs) - min(xs)) + margin * 2, (max(bs) - min(ys)) + margin * 2
    def rewrite(m):
        tag = m.group(0)
        tag = re.sub(r'\swidth="[^"]+"',   f' width="{bw:.3f}mm"', tag, count=1)
        tag = re.sub(r'\sheight="[^"]+"',  f' height="{bh:.3f}mm"', tag, count=1)
        tag = re.sub(r'\sviewBox="[^"]+"', f' viewBox="{bx:.3f} {by:.3f} {bw:.3f} {bh:.3f}"', tag, count=1)
        return tag
    return re.sub(r'<svg\b[^>]*>', rewrite, svg, count=1)


# Read a layout SVG, write its _interactive companion next to it
def make_interactive(src):
    dst = src.with_name(src.stem + "_interactive.svg")
    svg = src.read_text(encoding="utf8")
    svg = tighten_viewbox(svg)
    svg = re.sub(r'(<svg\b[^>]*\bxmlns="http://www\.w3\.org/2000/svg")',
                 r'\1 xmlns:xlink="http://www.w3.org/1999/xlink"', svg, count=1)
    svg = re.sub(r"(<svg\b[^>]*>)", r"\1" + STYLE_BLOCK, svg, count=1)
    svg, wrapped = wrap_groups(svg)
    dst.write_text(svg, encoding="utf8")
    return dst, wrapped


# Walk every layout_*.svg (excluding already-generated _interactive ones)
sources = sorted(p for p in LAYOUT_DIR.glob("layout_*.svg")
                 if not p.stem.endswith("_interactive"))
for src in sources:
    dst, n = make_interactive(src)
    print(f"  {src.name:35s} -> {dst.name}  ({n} boxes wrapped)")
print(f"Wrote {len(sources)} interactive layout SVG(s) into {LAYOUT_DIR.relative_to(REPO_ROOT)}.")

# FIN
