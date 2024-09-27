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

# Session Class
class Session:
    def __init__(self, readme_path=None):
        self.name = None            # name
        self.description = None     # description
        self.topics = None          # topics covered
        self.boxes = {}             # boxes opened dictionary {name:depth}
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

        # Extract name
        self.name = readme[line_count][2:-1]

        # Extract description
        self.description = []
        line_count = 1
        while readme[line_count][0] != '#':
            if readme[line_count][0:2] == '- ':
                # Extract boxes opened
                boxes_opened_text = readme[line_count][2:-1]
                self.boxes = self.parse_boxes_opened(boxes_opened_text)
            elif readme[line_count][0] != '\n':
                self.description.append(readme[line_count][:-1])
            line_count += 1
        self.description = "".join(self.description)

        # Extract topics
        self.topics = []
        while line_count < max_count:
            topic_text = []
            topic_text.append(readme[line_count][3:-1])
            line_count += 1
            while readme[line_count][0] != '#':
                if readme[line_count][0] != '\n':
                    topic_text.append(readme[line_count][:-1])
                line_count += 1
                if line_count >= max_count:
                    break
            topic = Topic.Topic(topic_text)
            self.topics.append(topic)
        return
    
    def parse_boxes_opened(self, boxes_opened_text):
        boxes_strings = boxes_opened_text.split(',')
        for box_string in boxes_strings:
            if box_string[0] == ' ':
                box_string = box_string[1:]
            box_name, box_level = box_string.split(' ')
            print(box_name, box_level)
        return "stuff"

    def render_topics(self, output_path):
        session_path = output_path + f'/{self.name.lower()}'
        Utilities.clear_folder(session_path)
        for t, topic in enumerate(self.topics):
            header = self.render_header(t)
            footer = self.render_footer(t)
            body = topic.render()
            output = header + body + footer
            topic_path = session_path + f'/{topic.name.replace(" ", "_").lower()}.html'
            with open(topic_path, "w") as file:
                file.write(output)
        return

    def render_header(self, topic_index):
        header = []
        header.append("<!DOCTYPE html>\n")
        header.append("<head>\n")
        header.append("{% include 'pwa.html' %}\n")
        header.append("<link rel=\"stylesheet\" type=\"text/css\" href=\"{{url_for('static', filename='styles/topic.css')}}\"/>\n")
        header.append("</head>\n\n")
        header.append("<html>\n<body>\n\n")
        header.append(f"<title>LBB : {self.name} : {self.topics[topic_index].name}</title>\n")
        header.append(f"<h2 id=\"session_name\">LBB : {self.name}</h2>\n")
        header.append(f"<h3 id=\"topic_name\">{self.topics[topic_index].name}</h3>\n")
        header.append(f"<h4 id=\"topic_description\">{self.topics[topic_index].description}</h4>\n")
        header.append(f"<hr>\n")
        return "".join(header)

    def render_footer(self, topic_index):
        footer = []
        footer.append("<hr>\n")
        if topic_index > 0:
            previous_topic_name = self.topics[topic_index-1].name.replace(" ", "_").lower()
            previous_topic = f"{previous_topic_name}"
            footer.append(f"<a href=\"{previous_topic}\">Previous Topic</a><br>\n")
        if topic_index < (len(self.topics)-1):
            next_topic_name = self.topics[topic_index+1].name.replace(" ", "_").lower()
            next_topic = f"{next_topic_name}"
            footer.append(f"<a href=\"{next_topic}\">Next Topic</a><br>\n")
        footer.append("</body>\n</html>")
        return "".join(footer)


#FIN