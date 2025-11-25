# -*- coding: utf-8 -*-
"""
LBB : Engine : Lesson Class

@author: kampff
"""

# Imports
import LBB.utilities as Utilities
import LBB.Engine.video as Video
import LBB.Engine.image as Image

# Lesson Class
class Lesson:
    """
    LBB Lesson Class

    Stores a link to a video tutorial (optional) and a list of steps to complete the lesson
    """ 
    def __init__(self, _session, box, text):
        self.course = _session.course   # Lesson parent (course)
        self.session = _session         # Lesson parent (session)
        self.box = box                  # Lesson box
        self.name = None                # Lesson name
        self.slug = None                # Lesson slug
        self.description = None         # Lesson description
        self.video = None               # Lesson video
        self.text = None                # Lesson text
        self.parse(text)                # Parse lesson from Markdown text
        return

    # Parse lesson string
    def parse(self, text):
        # Set line counter
        line_count = 0
        max_count = len(text)

        # Extract name and slug
        self.name, self.slug = Utilities.extract_lesson_name_and_slug(text[line_count])
        line_count += 1

        # Extract description
        self.description = text[line_count]
        line_count += 1

        # Extract video
        video_url = text[line_count].split('(')[1][:-1]
        if video_url != '':
            self.video = Video.Video(self, f"[{self.name}]({video_url})")
        line_count += 1

        # Find lesson section
        line_count = Utilities.find_line(text, "## Lesson")
        line_count += 1
        lesson_text = text[line_count:max_count]

        # Convert Markdown image links to HTML tags (centered with fixed width)
        parsed_text = []
        for line in lesson_text:
            image_link = Utilities.extract_markdown_image_link(line)
            if image_link:
                image = Image.Image(self, image_link)
                image_render_text = image.render()
                parsed_text.extend(image_render_text)
            else:
                parsed_text.append(line)

        # Extract lesson section
        self.text = parsed_text

        return

    # Render lesson object as Markdown
    def render(self):
        output = []
        if self.video:
            output.append(f"#### Watch this video: [{self.name}]({self.video.url})\n")
            output.append(f"{self.video.render()}\n")
        else:
            output.append(f"### {self.name}\n")
        output.append(f"> {self.description}\n\n")
        for line in self.text:
            output.append(line + '\n')
        output.append("\n")
        return output

#FIN