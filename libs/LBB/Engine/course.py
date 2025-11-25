# -*- coding: utf-8 -*-
"""
LBB : Engine : Course Class

@author: kampff
"""

# Imports
import glob
import LBB.config as Config
import LBB.utilities as Utilities
import LBB.Engine.session as Session

# Course Class
class Course:
    """
    LBB Course Class

    Stores a list course sessions
    """
    def __init__(self, name):
        self.name = None        # Course name
        self.slug = None        # Course slug
        self.sessions = None    # Course sessions
        self.build(name)        # Build course from repository
        return
    
    # Build course object from repository
    def build(self, name):
        """
        Build a version of an LBB course from the repository resources
        """
        # Set course parameters
        self.name = name
        self.slug = get_slug_from_name(name)

        # List session folders
        course_folder = f"{Config.course_path}/versions/{self.slug}"
        session_folders = sorted(glob.glob(f"{course_folder}/[0-9][0-9]_*"))
        
        # Load sessions from templates
        self.sessions = []
        for session_index, session_folder in enumerate(session_folders):
            session_path = f"{session_folder}/_resources/template.md"
            session_text = Utilities.read_clean_text(session_path)
            session = Session.Session(self, text=session_text)
            session.index = session_index
            self.sessions.append(session)
        return

# --------------
# Course Library
# --------------

# Get course slug from name
def get_slug_from_name(name):
    if name == "The Last Black Box":
        slug = "full"
    elif name == "Bootcamp":
        slug = "bootcamp"
    elif name == "Braitenberg":
        slug = "braitenberg"
    elif name == "Build a Brain":
        slug = "buildabrain"
    elif name == "AI-Workshops":
        slug = "ai-workshops"
    else:
        print("Unavailable course name selected!")
        exit(-1)
    return slug

#FIN