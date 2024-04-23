# -*- coding: utf-8 -*-
"""
LBB: User Class

@author: kampff
"""
#----------------------------------------------------------
# Load environment file and variables
import os
from dotenv import load_dotenv
load_dotenv()
libs_path = os.getenv('LIBS_PATH')
base_path = os.getenv('BASE_PATH')
data_path = base_path + '/_tmp'

# Set library paths
import sys
sys.path.append(libs_path)
#----------------------------------------------------------

# Import libraries
import os
import pickle
from werkzeug.security import generate_password_hash, check_password_hash

# Import modules

# User Class
class User:
    def __init__(self):
        self.id = None              # ID
        self.password_hash = None   # Password hash
        self.name = None            # Name
        self.email = None           # Email
        self.is_instructor = False  # Instructor boolean
        self.is_admin = False       # Administrator boolean
        self.authenticated = False  # Authenticated?
        return
    
    def is_active(self):
        """True, as all users are active."""
        return True

    def get_id(self):
        """Return the email address to satisfy Flask-Login's requirements."""
        return self.email

    def is_authenticated(self):
        """Return True if the user is authenticated."""
        return self.authenticated

    def is_anonymous(self):
        """False, as anonymous users aren't supported."""
        return False

    def store(self):
        user_data = {
            "id" : self.id,
            "password_hash" : self.password_hash,
            "name" : self.name,
            "email" : self.email,
            "is_instructor" : self.is_instructor,
            "is_admin" : self.is_admin,
        }
        user_folder = f"{data_path}/users/{self.id}"
        user_path = f"{user_folder}/user_data.pkl"
        with open(user_path, 'wb') as fp:
            pickle.dump(user_data, fp)

        return

# User helper functions

# Get user
def get(user_id):
    user_folder = f"{data_path}/users/{user_id}"
    user_path = f"{user_folder}/user_data.pkl"
    with open(user_path, 'rb') as pickle_file:
        user_data = pickle.load(pickle_file)
    print(user_data)
    #user = User.User()
    #user.id = "000000"
    return None

# FIN