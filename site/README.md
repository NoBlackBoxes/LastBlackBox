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

## To Do
Things to complete
- Download badge (certificate?)
- Send emails asynchronously
- Standardized task submissions (with validation and feedback)
- Student progress bars?
- Student topic/box completion state
- Discord integration?
- Run on S3?

## Model
Courses are composed of a sequence of boxes/depths. Each box is composed of a sequence of topics. Each topic is composed of a sequence of lessons covering up to 3 depths (level 01, 10, and 11).

```markdown
# Box Name

Box descrition.
Can be mutliple lines.

### Required Materials
- {01}: kits, tools, required, level 1
- {10}: kits, tools, required, level 2
- {11}: kits, tools, required, level 3

## Topic #1 Name

Topic description.
Just text.

{01}
Level 1 lesson for topic. Consisting of written instructions, images, videos, and tasks. Videos and tasks are defined in the following way.

- *Video*: [LBB:Box Name:Topic Name:Video Name](video url)

- **Task**: Task description.*(submission type)*

- *Video*: [LBB:Box Name:Topic Name:Video Name](video url)

- **Task**: Task description.*(submission type)*

{10}
Level 2 lesson for topic. Consisting of written instructions, images, videos, and tasks. Videos and tasks are defined in the following way.

- *Video*: [LBB:Box Name:Topic Name:Video Name](video url)

- **Task**: Task description.*(submission type)*

- *Video*: [LBB:Box Name:Topic Name:Video Name](video url)

- **Task**: Task description.*(submission type)*
```