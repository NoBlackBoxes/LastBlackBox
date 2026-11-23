# Course tools

Scripts for generating LBB course content, websites, designs, materials and media assets. Run commands from the repository root unless a tool's own README says otherwise.

## Complete website build

[`build_all.py`](build_all.py) is the CI/CD entry point:

```bash
python3 course/tools/build_all.py
```

It regenerates enabled course content, builds every published Docfx course site, and creates the root course catalogue, sitemap and `robots.txt`. Course selection and publishing are controlled by the central [course manifest](../courses.json).

See the [Docfx authoring and build guide](docfx/README.md) for installation, local preview, architecture and deployment details.

## Tool index

| Path | Purpose |
|---|---|
| [`engine/`](engine/) | Assemble session and box READMEs from course templates and reusable lessons. |
| [`docfx/`](docfx/) | Stage and build the static course websites, bundle assets and rewrite repository links. |
| [`materials/`](materials/) | Generate course bills of materials and packing lists, and update material records. |
| [`designs/`](designs/) | Generate shared course layouts, logos and icons. |
| [`packaging/`](packaging/) | Generate course-kit packaging dimensions, CAD models and print designs. |
| [`videos/`](videos/) | Generate the video catalogue, thumbnail frames and GIF previews. |
| [`renders/`](renders/) | Instructions for creating NB3 and component renders. |

## Generated files

These tools are generators, and several overwrite their outputs:

- `build_all.py` and `engine/generate_courses.py` rewrite generated session and box READMEs.
- `build_all.py` replaces the generated `docs/` website output.
- The design, materials, packaging and video tools write assets or data under `course/_resources/` or `boxes/`.

Review each specialist README and the script's configured output paths before running it. Check `git diff` afterwards so generated changes are intentional.
