# -*- coding: utf-8 -*-
"""
LBB: Course Class

@author: kampff
"""

# Import libraries
import glob
import json

# Import modules
import LBB.config as Config
import LBB.session as Session

# Course Class
class Course:
    def __init__(self, name=None, path=None):
        self.name = None        # Course name
        self.slug = None        # Course slug (URL)
        self.sessions = None    # Course sessions
        if name:
            self.build(name)    # Build course from session templates
        elif path:
            self.load(path)     # Load course from JSON file
        return

    # Convert course object to dictionary
    def to_dict(self):
        dictionary = {
            "name": self.name,
            "slug": self.slug,
            "sessions": [session.to_dict() for session in self.sessions]
        }
        return dictionary
    
    # Convert dictionary to course object
    def from_dict(self, dictionary):
        self.name = dictionary.get("name")
        self.slug = dictionary.get("slug")
        self.sessions = [Session.Session(dictionary=session_dictionary) for session_dictionary in dictionary.get("sessions", [])]
        return
    
    # Build course object from session templates
    def build(self, name):

        # Set course parameters
        self.name = name
        self.slug = get_slug_from_name(name)
        course_folder = f"{Config.course_root}/{self.slug}"
        
        # List session folders
        if self.slug == "course": # Full course
            session_folders = []
            for box_name in Config.box_names:
                session_folder = f"{Config.boxes_root}/{box_name.lower()}"
                session_folders.append(session_folder)
        else:
            session_folders = sorted(glob.glob(f"{course_folder}/session_*"))

        # Load sessions from templates
        self.sessions = []
        for session_index, session_folder in enumerate(session_folders):
            session_template = f"{session_folder}/template.md"
            with open(session_template, encoding='utf8') as f:
                lines = f.readlines()
            text = []
            for line in lines:
                if line.strip():                # Remove empty lines
                    text.append(line.rstrip())  # Remove trailing whitespace (including /n)
            session = Session.Session(text=text)
            session.index = session_index
            self.sessions.append(session)
        return

    # Load course object from JSON
    def load(self, path):
        with open(path, "r") as file:
            self.from_dict(json.load(file))
        return

    # Store course object in JSON file
    def store(self, path):
        with open(path, "w") as file:
            json.dump(self.to_dict(), file, indent=4)
        return

# Course Library

# Get course slug from name
def get_slug_from_name(name):
    if name == "The Last Black Box":
        slug = "course"
    elif name == "Bootcamp":
        slug = "bootcamp"
    elif name == "Build a Brain":
        slug = "buildabrain"
    else:
        print("Unavailable course name selected!")
        exit(-1)
    return slug

#FIN