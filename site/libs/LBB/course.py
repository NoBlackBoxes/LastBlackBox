# -*- coding: utf-8 -*-
"""
LBB: Course Class

@author: kampff
"""

# Import libraries
import os
import glob
import LBB.utilities as Utilities

# Import modules
import LBB.session as Session

# Course Class
class Course:
    def __init__(self, folder=None):
        self.name = None            # course name
        self.sessions = None        # sessions covered
        self.boxes = {}             # boxes opened dictionary {name:depth}
        if folder:
            self.load(folder)
        return

    def load(self, folder):
        # Extract course name
        folder_name = os.path.basename(folder)
        if folder_name == "bootcamp":
            self.name = "Bootcamp"
        elif folder_name == "buildabrain":
            self.name = "Build a Brain"
        else:
            print("Unavailable course selected!")
            exit(-1)
        # Load sessions
        session_folders = sorted(glob.glob(f"{folder}/session_*"))
        for session_folder in session_folders:
            session_readme = session_folder + "/README.md"
            session = Session.Session(session_readme)
            self.sessions.append(session)
            # Combine session boxes with course boxes
        
#FIN