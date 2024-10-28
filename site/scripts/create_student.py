# -*- coding: utf-8 -*-
"""
Scripts: Create Student

@author: kampff
"""

# Import libraries
from werkzeug.security import generate_password_hash
import LBB.utilities as Utilities

# Import modules
import LBB.config as Config
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
student.current_course = "Build a Brain"
student.course = Course.Course(name="Build a Brain")

# Create student data folder
student_folder = Config.data_root + f"/students/{student.id}"
Utilities.clear_folder(student_folder)

# Store student
student.store()

# Store course
student_courses_folder = Config.data_root + f"/students/{student.id}/courses"
course_path = student_courses_folder + f"/{student.course.slug}.json"
Utilities.clear_folder(student_courses_folder)
student.course.store(course_path)

# Generate and store student badge
student_badge_folder = Config.data_root + f"/students/{student.id}/badge"
Utilities.clear_folder(student_badge_folder)
student.generate_badge()

#FIN