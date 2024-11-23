# The Last Black Box Template Engine

## Setup
- Add library to "site-packages"
```bash
# On Host (current Python version 3.12.7, assuming LBB virtual environment)
echo "/home/${USER}/NoBlackBoxes/LastBlackBox/course/_engine/libs" >> /home/${USER}/NoBlackBoxes/LastBlackBox/_tmp/LBB/lib/python3.12/site-packages/LBB.pth
```

## Box Lesson Model
- Each **box** has multiple **lessons**, which are described in the "lessons.md" file
- Each **lesson** *may* have a corresponding **video** tutorial
- Each **lesson** *may* have multiple **instructions** and **tasks**
- Each **instruction** and **task** must specify a depth (difficulty) level (-, +, *)
- Each **instruction** consists of text, images, code, or *special* formats (notes, hints, help, challenges, etc.)
- Each **task** consists of a description, an optional set of **instructions**, and a **target**
- Each **target** consists of a single-line text description (can contain links)

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


