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
from werkzeug.security import generate_password_hash

# Import modules
import LBB.user as User

# Boxes status dictonary
boxes = {
    'Atoms'         : '01',
    'Electrons'     : '00',
    'Magnets'       : '00',
    'Light'         : '00',
    'Sensors'       : '00',
    'Motors'        : '00',
    'Transistors'   : '00',
    'Amplifiers'    : '01',
    'Circuits'      : '00',
    'Power'         : '00',
    'Data'          : '10',
    'Logic'         : '01',
    'Memory'        : '01',
    'FPGAs'         : '01',
    'Computers'     : '00',
    'Control'       : '00',
    'Robotics'      : '00',
    'Systems'       : '00',
    'Linux'         : '01',
    'Python'        : '10',
    'Networks'      : '10',
    'Websites'      : '01',
    'Servers'       : '01',
    'Security'      : '00',
    'Audio'         : '00',
    'Vision'        : '00',
    'Learning'      : '00',
    'Intelligence'  : '00'
}

# Define User
user = User.User()
user.id = "000011"
user.password_hash = generate_password_hash("4321")
user.name = "Jimmy Voight"
user.nickname = "Jimmy"
user.email = "info@voight-kampff.tech"
user.instructor = False
user.admin = False
user.boxes = boxes
user.current_course = 'build_a_brain'
user.current_box = 'electrons'
user.current_topic = 'measuring_voltage'

# Store User
user.store()

#FIN