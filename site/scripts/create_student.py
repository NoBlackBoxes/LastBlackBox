# -*- coding: utf-8 -*-
"""
Scripts: Create Student

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
import LBB.utilities as Utilities

# Import modules
import LBB.course as Course
import LBB.student as Student

# Progress dictionary
progress = {
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
    'Intelligence'  : '10'
}

# Define student
student = Student.Student()
student.id = "000011"
student.password_hash = generate_password_hash("1234")
student.name = "Jimmy Voight"
student.nickname = "Jimmy"
student.email = "info@voight-kampff.tech"
student.progress = progress
student.course = Course.Course(name="buildabrain")

# Create student data folder
student_folder = data_path + f"/students/{student.id}"
Utilities.clear_folder(student_folder)

# Store student
student.store()

#FIN