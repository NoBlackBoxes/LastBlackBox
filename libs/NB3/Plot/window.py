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
        self.handle = None

    def open(self):
        self.handle = pyglet.window.Window(self.width, self.height, caption=self.title, vsync=True, resizable=True)
        self.handle.switch_to()  # ensure GL context is current
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glLineWidth(1.0)

        self.batch = pyglet.graphics.Batch()

        @self.handle.event
        def on_resize(width, height):
            self.width, self.height = width, height
            return pyglet.event.EVENT_HANDLED

        @self.handle.event
        def on_close():
            self.handle.close()
            self.handle = None
            return pyglet.event.EVENT_HANDLED

    def process_events(self):
        if self.handle:
            self.handle.dispatch_events()

    def render(self):
        if self.handle:
            self.handle.clear()
            self.batch.draw()
            self.handle.flip()

    def close(self):
        if self.handle:
            self.handle.close()
            self.handle = None

# FIN
