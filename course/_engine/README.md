# The Last Black Box Template Engine

## Setup
- Add library to "site-packages"
```bash
# On Host (current Python version 3.12.3, assuming LBB virtual environment)
echo "/home/${USER}/NoBlackBoxes/LastBlackBox/course/_engine/libs" >> /home/${USER}/NoBlackBoxes/LastBlackBox/_tmp/LBB/lib/python3.12/site-packages/LBB.pth
```

## Box Model
- Each **course** is composed of multiple **sessions**, which cover the LBB material to a particular knowledge depth (01, 10, or 11).
- Each **session** opens a sequence of black **boxes** described in the session's *README* file.
- Each **box** presents a sequence of **lessons** consisting of a named video tutorial and text instructions.
- Each **lesson** might have one or more **tasks** for the student to complete in order to progress to the next **lesson**.
- Each **session** concludes with a **project**, the outcome of which the student must submit to certify completion.

```markdown
# Course Title : Session Number - Session Name
Session description.
Can be multiple lines, but just text.

## Box #1 Name
Box description.
Can be multiple lines, but just text.

#### Watch this video: [Video Title](video url)
> Video content description (single line of text).

#### Watch this video: [Video Title](video url)
> Video content description (single line of text).

- [ ] **Task**: Task description.
- Other task instructions: hints, help, warnings, challenges, etc.
<details><summary><strong>Target</strong></summary>
The expected results of completing the task are listed here. You can use images and links.
</details><hr>

## Box #2 Name
Can be multiple lines, but just text.

####  Watch this video: [Video Name](video url)
> Video content description (single line of text).

- **Task**: Task description.
- Other task instructions: hints, help, warnings, challenges, etc.<details><summary><strong>Target</strong></summary>
The expected results of completing the task are listed here. You can use images and links.
</details><hr>

- **Task**: Task description.
- Other task instructions: hints, help, warnings, challenges, etc.<details><summary><strong>Target</strong></summary>
The expected results of completing the task are listed here. You can use images and links.
</details><hr>

####  Watch this video: [Video Name](video url)
> Video content description (single line of text).
- **Task**: Task description.
- Other task instructions: hints, help, warnings, challenges, etc.
<details><summary><strong>Target</strong></summary>
The expected results of completing the task are listed here. You can use images and links.
</details><hr>

---

# Project
### Project Name
Session project description. Consisting of written instructions, images, videos, and links. Describe project goals, etc.
- Other project instructions: hints, help, warnings, challenges, etc.
- Other project instructions: hints, help, warnings, challenges, etc.

Submission instructions (link to Discord #channel)
```