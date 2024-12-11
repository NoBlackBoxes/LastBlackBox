# course : scripts : engine
The LBB template engine generates the session READMEs from a "template.md" file and the associated lessons and projects.

## Course Model
- Each **course** consists of multiple **sessions**


- Each **session** opens one or more black **boxes**course
- Each **box** has multiple **lessons**, which are located in the box's "_lessons" folder
- Each **lesson** *may* have a corresponding **video** tutorial
- Each **lesson** *may* have multiple **instructions** and **tasks**
- Each **instruction** and **task** must specify a depth (difficulty) level (-, +, *)
- Each **instruction** consists of text, images, code, or *special* formats (notes, hints, help, challenges, etc.)
- Each **task** consists of a description, an optional set of **instructions**, and a **target**
- Each **target** consists of a single-line text description (can contain links)

## Session Template
```markdown
# Title - Session Name
Session description. Single line of text.

## Box Name
Linkage text
{lesson1}
Linkage text
{lesson2}
...

## Box Name
Linkage text
{lesson1}
Linkage text
{lesson2}
...

# Project
Linkage text
{project1}
Linkage text
{lesson2}
...
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


