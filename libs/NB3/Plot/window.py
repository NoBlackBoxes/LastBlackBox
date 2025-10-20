# -*- coding: utf-8 -*-
"""
NB3 : Plot : Window Class

@author: kampff
"""

# Imports
import pyglet
from pyglet import gl

# Window Class
class Window:
    def __init__(self, width=640, height=480, title="NB3 Plot"):
        self.width = int(width)
        self.height = int(height)
        self.title = title
        self.batch = None
        self.window = None

    def open(self):
        self.window = pyglet.window.Window(self.width, self.height, caption=self.title, vsync=True, resizable=True)
        self.window.switch_to()  # ensure GL context is current
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glLineWidth(1.0)

        self.batch = pyglet.graphics.Batch()

        @self.window.event
        def on_resize(width, height):
            self.width, self.height = width, height
            return pyglet.event.EVENT_HANDLED

    def process_events(self):
        if self.window:
            self.window.dispatch_events()

    def render(self):
        """Draw the current frame NOW (no scheduling)."""
        if self.window:
            self.window.clear()
            self.batch.draw()
            self.window.flip()

    def close(self):
        if self.window:
            self.window.close()
            self.window = None

# FIN
