# -*- coding: utf-8 -*-
"""
LBB: Config Class

@author: kampff
"""

# Import libraries
import os
from dotenv import load_dotenv

# Load environment (.env) file
load_dotenv()

# Load Flask configuration variables
flask_secret_key = os.getenv('FLASK_SECRET_KEY')

# Load Protonmail configuration variables
protonmail_username = os.getenv('PROTONMAIL_USERNAME')
protonmail_smtp_token = os.getenv('PROTONMAIL_SMTP_TOKEN')

# Load LBB configuration variables
repo_root = os.getenv('REPO_ROOT')
course_root = repo_root + "/course"
site_root = repo_root + "/site"
data_root = site_root + "/_tmp"
static_root = site_root + "/static"

#FIN