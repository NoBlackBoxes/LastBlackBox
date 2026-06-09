# Authoring and publishing LBB course sites

This is the practical guide for working on the LBB course documentation sites: building them locally, adding new sessions, and shipping a new course.

The sites live at `https://noblackboxes.github.io/LastBlackBox/<slug>/` (e.g. [`/crick/`](https://noblackboxes.github.io/LastBlackBox/crick/)). They are produced by a two-stage pipeline:

1. The **LBB engine** ([libs/LBB/Engine/](../../../libs/LBB/Engine/)) reads `_resources/template.md` for every session, expands `{Box:Lesson}` tags into the relevant lesson content pulled from [boxes/](../../../boxes/), and writes the rendered `README.md` for each session.
2. **docfx** wraps those READMEs in a static site using a shared template under [template/](template/).

A push to `master` triggers [.github/workflows/deploy-docs.yml](../../../.github/workflows/deploy-docs.yml), which runs `build_all.py` and deploys the output to GitHub Pages.

---

## Local development

### One-time setup

Install the docfx CLI (requires the .NET 8 SDK):

```powershell
dotnet tool install docfx
```

`build_all.py` uses Python's stdlib only; no `pip install` step.

### Build everything

```powershell
uv run --no-project python course/tools/build_all.py
```

Output lands at `docs/<slug>/` (e.g. `docs/crick/`), with `docs/index.html` (landing page) and `docs/sitemap.xml` at the root. The script has five stages: interactive SVGs, engine READMEs, per-slug docfx sites, master sitemap, root assets. PYTHONPATH for the LBB import is set automatically.

`uv run --no-project` bypasses the Microsoft Store Python shim on Windows. On Linux / Mac, plain `python course/tools/build_all.py` is fine.

### Build just one course (faster iteration)

```powershell
uv run --no-project python course/tools/docfx/build_site.py crick
```

Skips Stage 0 (SVGs) and Stage 1 (engine READMEs); assumes those are already up to date. Use when only touching docfx config or template files for one course.

### Serve locally

```powershell
uv run --no-project python -m http.server -d docs 8000
```

Then [http://localhost:8000/](http://localhost:8000/). Stop with `Ctrl+C`.

---

## Authoring course content

### Anatomy of a session

A session lives at `course/versions/<slug>/[0-9][0-9]_<session-slug>/` and needs **one** hand-written file:

```
course/versions/<slug>/
  01_my-session/
    _resources/
      template.md       # hand-written; you edit this
    README.md           # auto-generated; do not edit by hand
```

`_resources/template.md` follows this shape:

```markdown
# Course Name : My Session

One-line description for the session.

## A topic

Body text, then a tag to embed a lesson:

{Electrons:Voltage}

More body text. Another tag:

{Sensors:Photodiodes}
```

The engine replaces each `{BoxName:LessonName}` tag with the full content of `boxes/<box>/_resources/lessons/<LessonName>.md`, plus any paired `.csv` materials file. The rendered `README.md` is written alongside `_resources/`.

### The `{Box:Lesson}` tag grammar

- **BoxName** — matches a folder under [boxes/](../../../boxes/), case-insensitive (`Electrons`, `Sensors`, `Audio`, etc.).
- **LessonName** — matches an `.md` file under `boxes/<box>/_resources/lessons/`. For example, `{Electrons:Voltage}` pulls `boxes/electrons/_resources/lessons/Voltage.md`.
- A matching `.csv` (e.g. `Voltage.csv` in the same dir) is treated as session-specific materials and merged into the per-session materials table.

You never modify the generated README directly; rebuild from the template.

### Regenerating session READMEs

After editing any `template.md`:

```powershell
uv run --no-project python course/tools/engine/generate_courses.py
```

Or just run `build_all.py`, which does this as Stage 1.

---

## Adding a new course

End-to-end procedure for a course slug that doesn't yet exist.

### 1. Author the session content

Create `course/versions/<slug>/[0-9][0-9]_<session-slug>/_resources/template.md` for each session. Sessions are sorted by their `NN_` numeric prefix.

### 2. Register the course with the engine

In [libs/LBB/config.py](../../../libs/LBB/config.py), add the course name to `course_names`:

```python
course_names = ["The Last Black Box", "Bootcamp", "Braitenberg", "Build a Brain", "Crick", "Your Course"]
```

In [libs/LBB/Engine/course.py](../../../libs/LBB/Engine/course.py), add the name→slug mapping in `get_slug_from_name`:

```python
elif name == "Your Course":
    slug = "your-course"
```

The slug is the directory name under `course/versions/`. Lowercase, hyphenated.

### 3. Author the docfx landing pages

A minimum docfx course needs two files at `course/versions/<slug>/`:

- **`index.md`** — the per-course landing page. Linked from the root site landing page. See [../../versions/crick/index.md](../../versions/crick/index.md) as a model.
- **`sessions.index.md`** — the sessions overview, linked from the top-nav. See [../../versions/crick/sessions.index.md](../../versions/crick/sessions.index.md).

Both are plain Markdown. Use repo-relative paths for images (`/course/_resources/...`) and session-relative paths for session links (`sessions/01_my-session/README.md`).

The defaults under [template/](template/) provide `docfx.json`, `sessions.toc.yml`, `toc.yml`, and the shared CSS/JS. Override any of those by placing a same-named file in `course/versions/<slug>/`.

### 4. Enable the docfx build

Add the slug to `SITES` in [../build_all.py](../build_all.py):

```python
SITES = ["crick", "your-course"]
```

### 5. Add a tile to the root landing page

Edit [root/index.html](root/index.html) — add a `<li>` inside `<ul class="sites">`:

```html
<li>
  <img class="site-icon" src="logo_yourcourse.png" alt="Your Course logo">
  <div class="body">
    <a href="your-course/">Your Course</a>
    <div class="blurb">One sentence describing what makes this course distinctive.</div>
    <a class="hint" href="your-course/">Open the course site &rarr;</a>
  </div>
</li>
```

Drop the icon image into `root/` alongside the existing logos. If you don't have a course icon, omit the `<img>` and adjust the `grid-template-columns` rule inline if needed.

### 6. Build, verify, ship

```powershell
uv run --no-project python course/tools/build_all.py
uv run --no-project python -m http.server -d docs 8000
```

Open [http://localhost:8000/your-course/](http://localhost:8000/your-course/) and confirm:

- Landing page renders with the right title and session links.
- Each session opens and its lesson tags expanded correctly.
- The root [http://localhost:8000/](http://localhost:8000/) shows your new tile and links into the course.

Then commit and push to `master`. CI deploys to Pages.

---

## CI deployment

[.github/workflows/deploy-docs.yml](../../../.github/workflows/deploy-docs.yml) runs `build_all.py` and publishes `docs/` to Pages on every push to `master`. The repo also needs:

- **Settings → Pages → Source = "GitHub Actions"** (one-time)
- **Settings → Environments → github-pages → Deployment branches** allows `master` (default). Add any other branch you want to deploy from while developing.

---

## Output anatomy

After a full build:

```
docs/
  index.html                # root landing page (course tiles)
  logo_*.svg, *.png         # root assets (from course/tools/docfx/root/)
  robots.txt                # root assets
  sitemap.xml               # master sitemap-index
  <slug>/                   # one subtree per course
    index.html              # course landing page (rendered from index.md)
    sitemap.xml             # per-course sitemap
    sessions/
      01_.../README.html
      02_.../README.html
      ...
    _assets/                # bundled images, etc.
    public/                 # docfx CSS/JS/search
```

Everything in `docs/` is generated. The directory is gitignored.

---

## File reference

| File | Hand-written? | Notes |
|---|---|---|
| `course/versions/<slug>/<NN>_*/_resources/template.md` | Yes | Session template with `{Box:Lesson}` tags. |
| `course/versions/<slug>/<NN>_*/README.md` | No | Generated by the engine. |
| `course/versions/<slug>/index.md` | Yes | Per-course landing page. |
| `course/versions/<slug>/sessions.index.md` | Yes | Sessions overview. |
| `course/versions/<slug>/docfx.json` | Optional | Override the template default. |
| `course/versions/<slug>/sessions.toc.yml` | Optional | Override the template default. |
| `libs/LBB/config.py` (`course_names`) | Yes | Register the course name. |
| `libs/LBB/Engine/course.py` (`get_slug_from_name`) | Yes | Map name to slug. |
| `course/tools/build_all.py` (`SITES`) | Yes | Enable the docfx build for the slug. |
| `course/tools/docfx/root/index.html` | Yes | Root landing page; add a tile per course. |
| `course/tools/docfx/root/*.svg`, `*.png` | Yes | Root assets, including per-course logos. |
| `course/tools/docfx/template/` | Rarely | Shared defaults: `docfx.json`, `toc.yml`, CSS/JS. |
