# -*- coding: utf-8 -*-
"""
LBB: Lesson Class

@author: kampff
"""

# Import libraries

# Import modules
import LBB.instruction as Instruction
import LBB.image as Image
import LBB.video as Video
import LBB.task as Task

# Lesson Class
class Lesson:
    def __init__(self, text=None):
        self.name = None            # lesson name
        self.description = None     # lesson description
        self.video = None           # lesson video
        self.steps = None           # lesson steps
        if text:
            self.parse(text)
        return

    def parse(self, text):
        # Set line counter
        line_count = 0
        max_count = len(text)

         # Extract name
        self.name = text[line_count].split('[')[1].split(']')[0]
        self.video = Video.Video(text[line_count])
        line_count += 1

        # Extract description
        self.description = text[line_count][:-1]
        line_count += 1

        # Extract steps
        self.steps = []
        max_count = len(text)
        while line_count < max_count:
            if text[line_count][0] != '\n':
                # Classify step
                if text[line_count][0:8] == '- **Task':
                    task = Task.Task(text[line_count])
                    self.steps.append(task)
                elif text[line_count][0:4] == '<img':
                    image = Image.Image(text[line_count])
                    self.steps.append(image)
                else:
                    instruction = Instruction.Instruction(text[line_count].strip())
                    self.steps.append(instruction)
            line_count += 1
        return

    def render(self, output_path):
        output = ''
        for step in self.steps:
            output = output + step.render()
        # join output
        # save file
        return output

    #def render(self, output_path, session_index):
    #    header = self.render_header(session_index)
    #    footer = self.render_footer(session_index)
    #    body = ''
    #    for box in self.boxes:
    #        body += box.render()
    #    project = self.project.render()
    #    output = header + body + project + footer
    #    with open(output_path, "w") as file:
    #        file.write(output)
    #    return
#
    #def render_header(self, session_index):
    #    header = []
    #    header.append("<!DOCTYPE html>\n")
    #    header.append("<head>\n")
    #    header.append("{% include 'pwa.html' %}\n")
    #    header.append("<link rel=\"stylesheet\" type=\"text/css\" href=\"{{url_for('static', filename='styles/session.css')}}\"/>\n")
    #    header.append("</head>\n\n")
    #    header.append("<html>\n<body>\n\n")
    #    header.append(f"<title>LBB : {self.name}</title>\n")
    #    header.append(f"<h2 id=\"session_name\">LBB : {self.name}</h2>\n")
    #    header.append(f"<hr>\n")
    #    return "".join(header)
#
    #def render_footer(self, session_index):
    #    footer = []
    #    footer.append("<hr>\n")
    #    footer.append("</body>\n</html>")
    #    return "".join(footer)

#FIN