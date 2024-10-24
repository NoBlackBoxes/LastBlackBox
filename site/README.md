# The Last Black Box Site

Instructions for building and maintaining the LBB site.

## Install Flask and Tools
```bash
pip install numpy flask flask-login flask-mail python-dotenv 
```

## Environment
You will need a .env file in the site root directory.

```bash
FLASK_APP=app
FLASK_ENV=development
FLASK_SECRET_KEY='a0ee1e54722175257d4aeb8d2b43ea5729f785450f69ac4b413652ef2e97ff36'
LIBS_PATH="/home/kampff/NoBlackBoxes/LastBlackBox/site/libs"
BASE_PATH="/home/kampff/NoBlackBoxes/LastBlackBox/site"
```

### Secret Key
Generate a secret key using the following command
```bash
python -c 'import secrets; print(secrets.token_hex())'
```
### Add Libraries to "site-packages"
```bash
# On Host (current Python version 3.12.3, assuming NBB virtual environment)
echo "/home/${USER}/NoBlackBoxes/LastBlackBox/site/libs" > /home/${USER}/NoBlackBoxes/LastBlackBox/_tmp/NBB/lib/python3.12/site-packages/LBB_site.pth
```

### Create "database" file system structure
```bash
# From site root
mkdir -p _tmp/users
```

## To Do
Things to complete
- *Course Model Revision*
- Download badge (certificate?)
- Send emails asynchronously
- Standardized task submissions (with validation and feedback)
- Student progress bars?
- Student box completion status
- Discord integration?
- Run on S3?

## Model
- Each **course** is composed of multiple **sessions**, which cover the LBB material to a particular knowledge depth (01, 10, or 11).
   - Depth 01: (LBB-10?) - *Build a Brain (101), Bootcamp (102), ENB (103), Own Phone (104)*
   - Depth 10: (LBB-20?) - *SWC-GCNU PhD* (201)
   - Depth 11: (LBB-30?) - *The Last Black Box* (301)
- Each **session** opens a sequence of black **boxes** described in the session's *README* file.
- Each **box** presents a sequence of **lessons** consisting of a named video tutorial and text instructions.
- Each **lesson** might have one or more **tasks** for the student to complete in order to progress to the next **lesson**.
- Each **session** concludes with a **project**, the outcome of which the student must submit to certify completion.

```markdown
# Course Title : Session Number - Session Name
Session description.
Can be multiple lines, but just text.

## Box #1 Name
Can be multiple lines, but just text.

### Watch this video: [Video Title](video url)
Video content description (single line of text).

### Watch this video: [Video Title](video url)
Video content description (single line of text).
- **Task**(task_name): Task description.[task_input]

## Box #2 Name
Can be multiple lines, but just text.

###  Watch this video: [Video Name](video url)
Video content description (single line of text).
- **Task**(task_name): Task description.[task_input]
- **Task**(task_name): Task description.[task_input]

###  Watch this video: [Video Name](video url)
Video content description (single line of text).
- **Task**(task_name): Task description.[task_input]

---

# Project
### Project Name
Session project description. Consisting of written instructions, images, videos, and links. Describe project goals, etc.

Submission instructions (link to Discord #channel)
```