# -*- coding: utf-8 -*-
"""
LBB: User Class

@author: kampff
"""

# Import libraries
import os
from werkzeug.security import generate_password_hash, check_password_hash

# Import modules

# User Class
class User:
    def __init__(self):
        self.id = None              # ID
        self.name = None            # name
        self.is_instructor = False  # instructor boolean
        self.is_admin = False       # instructor boolean
        self.email = None
        self.authenticated = False
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


# User helper functions

# Get user
def get(user_id):
    user = User()
    user.id = 000000
    return user

# FIN