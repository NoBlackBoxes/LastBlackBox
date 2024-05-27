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
import pickle
from werkzeug.security import generate_password_hash, check_password_hash

# Import modules

# User Class
class User:
    def __init__(self, _user_id=None):
        self.id = None              # ID
        self.password_hash = None   # Password hash
        self.name = None            # Name
        self.email = None           # Email
        self.instructor = False     # Instructor boolean
        self.admin = False          # Administrator boolean
        self.authenticated = False  # Authenticated boolean
        self.loaded = False         # Loaded boolean
        if (_user_id != None):
            self.load(user_id=_user_id)
        return
    
    def is_active(self):
        return True

    def get_id(self):
        return self.id

    def is_authenticated(self):
        return self.authenticated

    def is_anonymous(self):
        return False

    def store(self):
        user_data = {
            "id" : self.id,
            "password_hash" : self.password_hash,
            "name" : self.name,
            "email" : self.email,
            "instructor" : self.instructor,
            "admin" : self.admin,
        }
        user_folder = f"{data_path}/users/{self.id}"
        user_path = f"{user_folder}/user_data.pkl"
        with open(user_path, 'wb') as fp:
            pickle.dump(user_data, fp)
        return

    def load(self, user_id):
        user_folder = f"{data_path}/users/{user_id}"
        user_path = f"{user_folder}/user_data.pkl"
        # Does user (data file) exist?
        if not os.path.isfile(user_path):
            return self
        # Load user data
        with open(user_path, 'rb') as pickle_file:
            user_data = pickle.load(pickle_file)
        self.id = user_data['id']
        self.password_hash = user_data['password_hash']
        self.name = user_data['name']
        self.email = user_data['email']
        self.instructor = user_data['instructor']
        self.admin = user_data['admin']
        self.loaded = True
        print(user_data)
        return

# FIN