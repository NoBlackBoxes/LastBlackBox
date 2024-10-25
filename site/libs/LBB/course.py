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
    def __init__(self, course_folder=None):
        self.name = None            # course name
        self.slug = None            # course slug (URL)
        self.sessions = None        # course sessions
        self.depth = None           # course depth
        if course_folder:
            self.load(course_folder)
        return

    def load(self, folder):
        # Extract course name and slug
        folder_name = os.path.basename(folder)
        if folder_name == "bootcamp":
            self.name = "Bootcamp"
            self.slug = "bootcamp"
            self.depth = 1
        elif folder_name == "buildabrain":
            self.name = "Build a Brain"
            self.slug = "buildabrain"
            self.depth = 1
        else:
            print("Unavailable course selected!")
            exit(-1)
        
        # Load sessions
        self.sessions = []
        session_folders = sorted(glob.glob(f"{folder}/session_*"))
        for session_folder in session_folders:
            session_readme = session_folder + "/README.md"
            session = Session.Session(session_readme)
            self.sessions.append(session)
        return

    def render(self, template_folder):
        # Render schedule
        schedule_template_path = template_folder + f"/schedule.html"
        self.render_schedule(schedule_template_path)
        
        # Render each session
        for s, session in enumerate(self.sessions):
            session_template_folder = template_folder + f"/session_{s}"
            Utilities.clear_folder(session_template_folder)

            # Render each box
            for box in session.boxes:
                box_template_folder = session_template_folder + f"/{box.name.lower()}"
                Utilities.clear_folder(box_template_folder)

                # Render each lesson
                for lesson in box.lessons:
                    lesson_url = lesson.name.lower().replace(' ', '-').replace('\'', '') + ".html"
                    lesson_template_path = box_template_folder + f"/{lesson_url}"
                    header = self.render_header(session, box, lesson)
                    footer = self.render_footer(session, box, lesson)
                    body = lesson.render()
                    output = header + body + footer
                    with open(lesson_template_path, "w") as file:
                        file.write(output)
                    print(f"Rendered Template: {self.slug}/session_{s}/{box.name.lower()}/{lesson_url}")
        return

    def render_header(self, session, box, lesson):
        header = []
        header.append("<!DOCTYPE html>\n")
        header.append("<head>\n")
        header.append("{% include 'pwa.html' %}\n")
        header.append("<link rel=\"stylesheet\" type=\"text/css\" href=\"{{url_for('static', filename='styles/lesson.css')}}\"/>\n")
        header.append("</head>\n\n")
        header.append("<html>\n<body>\n\n")
        header.append(f"<title>LBB : {self.name}</title>\n")
        header.append(f"<h3 id=\"course_heading\">{self.name} : {session.name} : {box.name} : <span id=\"lesson_name\">{lesson.name}</span></h3>\n")
        header.append(f"<hr>\n")
        return "".join(header)
    
    def render_footer(self, session, box, lesson):
        footer = []
        footer.append("<hr>\n")
        footer.append("</body>\n</html>")
        return "".join(footer)

    def render_schedule(self, schedule_template_path):
        header = []
        header.append("<!DOCTYPE html>\n")
        header.append("<head>\n")
        header.append("{% include 'pwa.html' %}\n")
        header.append("<link rel=\"stylesheet\" type=\"text/css\" href=\"{{url_for('static', filename='styles/lesson.css')}}\"/>\n")
        header.append("</head>\n\n")
        header.append("<html>\n<body>\n\n")
        header.append(f"<title>LBB : {self.name}</title>\n")
        header.append(f"<h3 id=\"course_heading\">{self.name}</h3>\n")
        header.append(f"<hr>\n")
        header = "".join(header)
        body = "<a href=\"/buildabrain/session_0/atoms/atomic-structure.html\">Lesson Example</a>\n"
        footer = []
        footer.append("<hr>\n")
        footer.append("</body>\n</html>")
        footer = "".join(footer)
        output = header + body + footer
        with open(schedule_template_path, "w") as file:
            file.write(output)
        print(f"Rendered Template: {self.slug}/schedule.html")
        return

#FIN