# -*- coding: utf-8 -*-
"""
LBB: Session Class

@author: kampff
"""

# Import libraries
import os
import LBB.utilities as Utilities

# Import modules
import LBB.topic as Topic
import LBB.project as Project

# Session Class
class Session:
    def __init__(self, readme_path=None):
        self.name = None            # session name
        self.description = None     # session description
        self.boxes = {}             # boxes opened dictionary {name:depth}
        self.topics = None          # topics covered
        self.project = None         # session project
        if readme_path:
            self.parse_readme(readme_path)
        return
    
    def parse_readme(self, readme_path):
        # Read README.md
        with open(readme_path, encoding='utf8') as f:
            readme = f.readlines()

        # Line counter
        line_count = 0
        max_count = len(readme)

        # Extract session name
        title = readme[line_count][2:-1]
        self.name = title.split('-')[1][1:]

        # Extract session description
        self.description = []
        line_count = 1
        while readme[line_count][0] != '#':
            if readme[line_count][0:2] == '- ':
                # Extract boxes opened
                self.boxes = {}
                boxes_opened_text = readme[line_count][2:-1]
                boxes_strings = boxes_opened_text.split(',')
                for box_string in boxes_strings:
                    if box_string[0] == ' ':
                        box_string = box_string[1:]
                    name, level = box_string.split(' ')
                    self.boxes.update({name:level[1:-1]})
            elif readme[line_count][0] != '\n':
                self.description.append(readme[line_count][:-1])
            line_count += 1
        self.description = "".join(self.description)

        # Extract session topics
        self.topics = []
        while readme[line_count][0:3] != '---':
            line_count += 1
        topic_line_count = line_count
        line_count = 1
        while line_count < topic_line_count:
            topic_text = []
            topic_text.append(readme[line_count][3:-1])
            line_count += 1
            while readme[line_count][0] != '#':
                if readme[line_count][0] != '\n':
                    topic_text.append(readme[line_count][:-1])
                line_count += 1
                if line_count >= topic_line_count:
                    break
            topic = Topic.Topic(topic_text)
            self.topics.append(topic)
        line_count += 2
    
        # Extract session project
        while line_count < max_count:
            project_text = []
            project_text.append(readme[line_count][2:-1])
            line_count += 1
            while readme[line_count][0] != '#':
                if readme[line_count][0] != '\n':
                    project_text.append(readme[line_count][:-1])
                line_count += 1
                if line_count >= max_count:
                    break
        self.project = Project.Project(project_text)
        return

    def render(self, output_folder, session_index):
        session_path = output_folder + f'/{self.name.replace(" ", "_").lower()}.html'
        header = self.render_header(session_index)
        footer = self.render_footer(session_index)
        body = ''
        for topic in self.topics:
            body += topic.render()
        project = self.project.render()
        output = header + body + project + footer
        with open(session_path, "w") as file:
            file.write(output)
        return

    def render_header(self, session_index):
        header = []
        header.append("<!DOCTYPE html>\n")
        header.append("<head>\n")
        header.append("{% include 'pwa.html' %}\n")
        header.append("<link rel=\"stylesheet\" type=\"text/css\" href=\"{{url_for('static', filename='styles/topic.css')}}\"/>\n")
        header.append("</head>\n\n")
        header.append("<html>\n<body>\n\n")
        header.append(f"<title>LBB : {self.name}</title>\n")
        header.append(f"<h2 id=\"session_name\">LBB : {self.name}</h2>\n")
        header.append(f"<hr>\n")
        return "".join(header)

    def render_footer(self, session_index):
        footer = []
        footer.append("<hr>\n")
        footer.append("</body>\n</html>")
        return "".join(footer)


#FIN