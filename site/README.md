# lastbackbox.training

This folder contains the front- and back-end tools and code from creating the Last Black Box website.

----

## Prerequisites

We will use a python web framework called Flask to build and manage the LBB website.

- Create a python virtual environment (requires Python version > 3.4)

```bash
cd $LBBROOT/repo/site
python3 -m venv venv
```

- Activate the virtual environment

```bash
source venv/bin/activate
```

- Install flask

```bash
pip install flask
```

- Install dotenv (for storing environment variables)

```bash
pip install python-dotenv
```

- Create a .flaskenv file in the site root folder and add the LBB application:

```bash
FLASK_APP=lbb.py
```

- Install Flask web form toolkit

```bash
pip install flask-wtf
```


# WTF

Just hosted this on AWS

Lightsail

get SSH key, put somwwhere

can use scp to copy site

run python script using INTERNAL IP to bind port

then connect via browser to external IP (listed in Amazon site)

whoa

....!

this is crazy