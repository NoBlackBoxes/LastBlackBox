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
- Each **instruction** and **task** must specify a depth (difficulty) level (-, +, *)
- Each **instruction** consists of text, images, code, or *special* formats (notes, hints, help, challenges, etc.)
- Each **task** consists of a description, an optional set of **instructions**, and a **target**
- Each **target** consists of a single-line text description (can contain links) of the expected task outcome

## Course Hierarchy
- Course
  - Sessions (*template.md*)
    - Box #1
      - Materials (*materials.csv*)
      - Lessons (*lessons.md*)
    - Box #2
      - Materials (*materials.csv*)
      - Lessons (*lessons.md*)
    - ...
    - Project

## Session Template (template.md)
```markdown
# Course Title - Session Name
Session description. Single line of text.

# Example Main Heading
---
## Example Sub-Heading (Box Name)
{materials:box:depth}
Linkage text
{lesson:box:name:depth}
Linkage text
{lesson:box:name:depth}

## Example Sub-Heading (Box Name)
{materials:box:depth}
Linkage text
{lesson:box:name:depth}
Linkage text
{lesson:box:name:depth}

# Project
Linkage text
{project:box:name:depth}
```

## Lesson Template
```markdown
# Box Name
Box description. Single line of text.

## [Lesson Name](<video-url>)
> Lesson description. Single line of text

- Level 1 instruction text
- ![Level 1 instruction image:width](<image path or url>)
+ Level 2 instruction text
+ ![Level 2 instruction image:width](<image path or url>)
* Level 3 instruction text
* ![Level 3 instruction image:width](<image path or url>)

- **TASK**Task: Level 1 task description
  - Task instruction text
  - ![Task instruction image](<image path or url>)
> Task Target. Single line of text.

+ Level 2 instruction text
+ ![Level 2 instruction image:width](<image path or url>)

+ **TASK**Task: Level 2 task description
  - Task text
  - ![Task instruction image](<image path or url>)
> Task Target. Single line of text.

* Level 3 instruction text
* ![Level 3 instruction image:width](<image path or url>)

* **TASK**Task: Level 3 task description
  - Task text
  - ![Task instruction image](<image path or url>)
> Task Target. Single line of text.

## [Lesson Name](<video-url>)
> Lesson description. Single line of text

- Level 1 instruction text
- ![Level 1 instruction image:width](<image path or url>)
+ Level 2 instruction text
* Level 3 instruction text

- **TASK**Task: Level 1 task description
  - Task instruction text
  - ![Task instruction image](<image path or url>)
> Task Target. Single line of text.

+ Level 2 instruction text
+ ![Level 2 instruction image:width](<image path or url>)

+ **TASK**Task: Level 2 task description
  - Task text
  - ![Task instruction image](<image path or url>)
> Task Target. Single line of text.

* Level 3 instruction text
* ![Level 3 instruction image:width](<image path or url>)

* **TASK**Task: Level 3 task description
  - Task text
  - ![Task instruction image](<image path or url>)
> Task Target. Single line of text.
```


