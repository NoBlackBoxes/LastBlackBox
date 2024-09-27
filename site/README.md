# Site

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
# On Host (current Python version 3.12.3)
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
- Student topic/box completion state
- Discord integration?
- Run on S3?

## Model
Courses open a sequence of "black boxes" that cover the material to a particular depth.
 - Depth 01: (LBB-101) - *Bootcamp, ENB, Build a Brain, Own Phone*
 - Depth 10: (LBB-201) - *The Last Black Box*
 - Depth 11: (LBB-301) - *The Last Black Box (Advanced)*

Each session is composed of a sequence of topics and described in a "README.md" contained in the numbered session folders of each course. Each topic is composed of a sequence of lessons with tasks for the student to complete in order to progress to the next lesson.

```markdown
# Course Title : Session Number - Name

Session descrition.
Can be mutliple lines.
- first box opened (depth), next box opened (depth), ...

## Topic #1 Name

Topic description.
Just text.

First lesson for topic. Consisting of written instructions, images, videos, and tasks. Videos and tasks are defined in the following way.

- *Watch this video*: [Video Name](video url)
  - ***Task***(task name): Task description [submission type].

- *Watch this video*: [Video Name](video url)
  - **Task**(task name): Task description [submission type].

Second lesson for topic. Consisting of written instructions, images, videos, and tasks. Videos and tasks are defined in the following way.

---

# Project

Session project description.

```