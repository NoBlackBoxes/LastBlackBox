# -*- coding: utf-8 -*-
"""
NB3 : Plot : Line Class

@author: kampff
"""

# Imports
import numpy as np
import pyglet
from pyglet import gl
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.graphics import ShaderGroup

# Define shaders
vs = """
#version 330
uniform vec2 u_screen;
in vec2 position;
void main(){
    vec2 ndc = (position / u_screen) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    gl_Position = vec4(ndc, 0.0, 1.0);
}"""
fs = """
#version 330
out vec4 fragColor;
void main(){ fragColor = vec4(1,1,0,1); }"""

# Line Class
class Line:
    def __init__(self, width, height, min, max, num_samples):
        self.window = None
        self.program = None
        self.group = None
        self.batch = None
        self.vertex_list = None
        self.width = int(width)
        self.height = int(height)
        self.min = min
        self.max = max
        self.num_samples = int(num_samples)
        self.vertices = None  # interleaved x,y (float32)

    def open(self):
        """Create window + GL objects. Call from the main thread."""
        self.window = pyglet.window.Window(self.width, self.height, caption="NB3 Line Plot", vsync=True)
        self.window.switch_to()  # ensure GL context is current
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glLineWidth(1.0)

        self.program = ShaderProgram(Shader(vs, "vertex"), Shader(fs, "fragment"))
        self.program["u_screen"] = (self.width, self.height)
        self.group = ShaderGroup(self.program)

        # Interleaved vertex buffer: x across width, y initially mid-line
        x = np.linspace(0, self.width - 1, self.num_samples, dtype=np.float32)
        self.vertices = np.empty(2 * self.num_samples, dtype=np.float32)
        self.vertices[0::2] = x
        self.vertices[1::2] = (self.max - self.min) * 0.5  # mid

        self.batch = pyglet.graphics.Batch()
        self.vertex_list = self.program.vertex_list(
            self.num_samples, gl.GL_LINE_STRIP, batch=self.batch, group=self.group,
            position=("f", self.vertices)
        )

        @self.window.event
        def on_resize(width, height):
            self.width, self.height = width, height
            self.program["u_screen"] = (width, height)
            # keep x spanning new width
            self.vertices[0::2] = np.linspace(0, width - 1, self.num_samples, dtype=np.float32)
            self.vertex_list.position[:] = self.vertices
            return pyglet.event.EVENT_HANDLED

    def process_events(self):
        """Pump window events so the OS doesn't mark the window unresponsive."""
        if self.window:
            self.window.dispatch_events()

    def draw_data(self, buffer: np.ndarray):
        y = np.asarray(buffer, dtype=np.float32)
        if y.size != self.num_samples:
            # If sizes differ, take the tail or pad on the left with last value.
            if y.size > self.num_samples:
                y = y[-self.num_samples:]
            else:
                pad = np.full(self.num_samples - y.size, y[-1] if y.size else 0, np.uint8)
                y = np.concatenate([pad, y])
        # map to pixel Y and push to GPU (update only Ys = odd indices)
        self.vertices[1::2] = y.astype(np.float32) * 1000.0 + 128
        self.vertex_list.position[:] = self.vertices

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
