# course : tools : engine
The template engine generates a courses's **session** READMEs from a "template.md" file and the associated **lessons** (stored in a "Lesson-Name.md" file).

## Course Model
- Each **course** consists of multiple (numbered) **session** folders (e.g. 01_intro/)
  - **sessions** are defined in a *template.md* file listing a sequence of **lessons**
- Each **lesson** is linked to a "black box" in the main "boxes" section of the LBB repo
  - **lessons** are defined in a *Lesson-Name.md* file in the box's "_resources/lessons" folder
  - **lessons** *may* have a corresponding **video** tutorial
  - **lessons** *may* have a "Lesson-Name.csv" that lists additional materials required for that lesson
    - The **materials** required to complete each **session** are the combination of the *core* materials in each opened box's "materials.csv" file and these additional *lesson-specific* materials
