# -*- coding: utf-8 -*-
"""
Scripts: Create User

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
from werkzeug.security import generate_password_hash, check_password_hash

# Import modules
import LBB.user as User

# Define User
user = User.User()
user.id = "000000"
user.password_hash = generate_password_hash("1234")
user.name = "Adam Kampff"
user.email = "kampff@voight-kampff.tech"
user.instructor = True
user.admin = True

# Store User
user.store()

#FIN