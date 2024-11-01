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
        self.depth = None       # Course depth
        self.sessions = None    # Course sessions
        if name:
            self.build(name)    # Build course from session READMEs
        elif path:
            self.load(path)     # Load course from JSON file
        return

    # Convert course object to dictionary
    def to_dict(self):
        dictionary = {
            "name": self.name,
            "slug": self.slug,
            "depth": self.depth,
            "sessions": [session.to_dict() for session in self.sessions]
        }
        return dictionary
    
    # Convert dictionary to course object
    def from_dict(self, dictionary):
        self.name = dictionary.get("name")
        self.slug = dictionary.get("slug")
        self.depth = dictionary.get("depth")
        self.sessions = [Session.Session(dictionary=session_dictionary) for session_dictionary in dictionary.get("sessions", [])]
        return
    
    # Build course object from session README files
    def build(self, name):
        # Set course name, slug, and depth
        self.name = name
        if self.name == "Bootcamp":
            self.slug = "bootcamp"
            self.depth = 1
        elif self.name == "Build a Brain":
            self.slug = "buildabrain"
            self.depth = 1
        else:
            print("Unavailable course selected for build!")
            exit(-1)

        # Set course folder
        course_folder = f"{Config.course_root}/{self.slug}"
        
        # Load sessions from READMEs
        self.sessions = []
        session_folders = sorted(glob.glob(f"{course_folder}/session_*"))
        for session_index, session_folder in enumerate(session_folders):
            session_readme = session_folder + "/README.md"
            with open(session_readme, encoding='utf8') as f:
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

#FIN