# The Last Black Box Template Engine

## Setup
- Add library to "site-packages"
```bash
# On Host (current Python version 3.12.3, assuming LBB virtual environment)
echo "/home/${USER}/NoBlackBoxes/LastBlackBox/course/_engine/libs" >> /home/${USER}/NoBlackBoxes/LastBlackBox/_tmp/LBB/lib/python3.12/site-packages/LBB.pth
```

## Box Lesson Model
- Each **box** has multiple **lessons**, which are described in the "lessons.md" file
- Each **lesson** may have a corresponding **video** tutorial
- Each **lesson** has multiple **instructions** and **tasks**
- Each **instruction** and **task** specifies a particular depth (difficulty) level (-, +, *)
- Each **instruction** can consist of plain text, an image, or *special* formats (notes, hints, help, challenges, etc.)
- Each **task** consists of a description, an optional set of **instructions**, and a **target**
- Each **target** consists of a description and an optional set of **instructions**

```markdown
# Box Name
Box description. Single line of text.

## [Lesson Name](<video-url>)
> Lesson description. Single line of text

- Level 1 instruction text

- **TASK**Task: Level 2 description
  - Task text
> Task Target

+ Level 2 instruction

+ **TASK**Task: Level 2 Task description
  - Task text
> Task Target
```


