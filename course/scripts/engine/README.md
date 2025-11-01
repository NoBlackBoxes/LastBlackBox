# course : scripts : engine
The LBB template engine generates the session READMEs from a "template.md" file and the associated lessons and projects.

## Course Model
- Each **course** consists of multiple **sessions**
- Each **session** outlines a sequence of black **boxes** to open and a **project**
  - **Sessions** are defined in a *template.md* file
- Each **box** contains the **materials** and **lessons** required to open the box
- Each **project** describes the **tasks** to complete the session
- Each **lesson** *may* have a corresponding **video** tutorial
- Each **lesson** *may* have multiple **instructions** and **tasks**
  - **Lessons** are defined in a *lessons.md* file
- Each **instruction** consists of text, images, code, or *special* formats (notes, hints, help, challenges, etc.)
- Each **task** consists of a description, an optional set of **instructions**, and a **target**
- Each **target** consists of a single-line text description (can contain links) of the expected task outcome

## Course Hierarchy
- Course
  - Session #1 (*template.md*)
    - Box #1
      - Materials (*materials.csv*)
      - Lessons (*lessons.md*)
    - Box #2
      - Materials (*materials.csv*)
      - Lessons (*lessons.md*)
    - ...
    - Project
  - Session #2 (*template.md*)
    - Box #3
      - Materials (*materials.csv*)
      - Lessons (*lessons.md*)
    - Box #4
      - Materials (*materials.csv*)
      - Lessons (*lessons.md*)
    - ...
    - Project



