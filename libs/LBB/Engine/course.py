# -*- coding: utf-8 -*-
"""
LBB : Engine : Course Class

@author: kampff
"""

# Imports
import glob
import json
import LBB.utilities as Utilities
import LBB.config as Config
import LBB.Engine.session as Session

# Course Class
class Course:
    """
    LBB Course Class

    Stores a list course sessions and boxes
    """
    def __init__(self, name=None, path=None):
        self.name = None        # Course name
        self.slug = None        # Course slug (URL)
        self.sessions = None    # Course sessions
        if name:
            self.build(name)    # Build course from repository
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
    
    # Build course object from repository
    def build(self, name):
        """
        Build a version of the LBB course from the repository resources
        """
        # Set course parameters
        self.name = name
        self.slug = get_slug_from_name(name)
        
        # List session folders
        if self.slug == "course": # Full course
            session_folders = []
            for box_name in Config.box_names:
                session_folder = f"{Config.boxes_root}/{box_name.lower()}"
                session_folders.append(session_folder)
        else:
            course_folder = f"{Config.course_root}/versions/{self.slug}"
            session_folders = sorted(glob.glob(f"{course_folder}/[0-9][0-9]_*"))

        # Load sessions from templates
        self.sessions = []
        for session_index, session_folder in enumerate(session_folders):
            session_path = f"{session_folder}/_resources/template.md"
            session_text = Utilities.read_clean_text(session_path)
            session = Session.Session(text=session_text)
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

# --------------
# Course Library
# --------------

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