# -*- coding: utf-8 -*-
"""
NB3 : Plot : Axes Class

@author: kampff
"""

# Imports
import pyglet
from pyglet import gl
from pyglet import shapes

# Axes Class
class Axes:
    def __init__(self, width=640, height=480, title="NB3 Plot"):
        self.width = int(width)
        self.height = int(height)
        self.title = title
        self.batch = None
        self.window = None
        self.x_axis = None
        self.y_axis = None

    def open(self):
        self.window = pyglet.window.Window(self.width, self.height, caption=self.title, vsync=True, resizable=True)
        self.window.switch_to()
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glLineWidth(1.0)
        self.batch = pyglet.graphics.Batch()
        self._draw_coords()

        @self.window.event
        def on_resize(width, height):
            self.width, self.height = width, height
            self._draw_coords()
            return pyglet.event.EVENT_HANDLED

        @self.window.event
        def on_close():
            self.window.close()
            self.window = None
            return pyglet.event.EVENT_HANDLED

    def process_events(self):
        if self.window:
            self.window.dispatch_events()

    def render(self):
        if self.window:
            self.window.clear()
            self.batch.draw()
            self.window.flip()

    def close(self):
        if self.window:
            self.window.close()
            self.window = None

    def _draw_coords(self):
        x0 = self.width/2.0
        y0 = self.height/2.0
        self.x_axis = shapes.Line(0, y0, self.width, y0, thickness=2, color=(64, 64, 64), batch=self.batch)
        self.y_axis = shapes.Line(x0, 0, x0, self.height, thickness=2, color=(64, 64, 64), batch=self.batch)

#FIN
