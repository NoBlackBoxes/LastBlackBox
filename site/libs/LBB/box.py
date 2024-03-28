# -*- coding: utf-8 -*-
"""
LBB: Box Class

@author: kampff
"""

# Import libraries
import os

# Import modules
import LBB.topic as Topic
import LBB.utilities as Utilities

# Box Class
class Box:
    def __init__(self, readme_path=None):
        self.name = None            # name
        self.description = None     # description
        self.materials = None       # required materials
        self.topics = None          # topics list
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
            if readme[line_count][0] != '\n':
                self.description.append(readme[line_count][:-1])
            line_count += 1
        self.description = "".join(self.description)

        # Extract material lists for each depth (01,10,11)
        self.materials = []
        line_count += 1
        while readme[line_count][0] != '#':
            if readme[line_count][0] != '\n':
                self.materials.append(readme[line_count][8:-1])
            line_count += 1

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

    def render_topics(self, output_path):
        box_path = output_path + f'/{self.name.lower()}'
        Utilities.clear_folder(box_path)
        for t, topic in enumerate(self.topics):
            header = Utilities.render_header(self, t)
            footer = Utilities.render_footer(self, t)
            body = topic.render()
            output = header + body + footer
            topic_path = box_path + f'/{topic.name.replace(" ", "_").lower()}.html'
            with open(topic_path, "w") as file:
                file.write(output)

#FIN