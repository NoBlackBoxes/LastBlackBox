# -*- coding: utf-8 -*-
"""
NB3 : Plot : Line Class

@author: kampff
"""

# Imports
import numpy as np
import NB3.Plot.window as Window
import pyglet
from pyglet import gl
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.graphics import ShaderGroup

# Define shaders
vs = """
#version 330
uniform vec2 screen_dim;
in vec2 position;
void main(){
    vec2 ndc;
    ndc.x = (position.x / screen_dim.x) * 2.0 - 1.0;
    ndc.y = position.y;
    gl_Position = vec4(ndc, 0.0, 1.0);
}"""
fs = """
#version 330
out vec4 fragColor;
void main(){ fragColor = vec4(1,1,0,1); }"""

# Line Class
class Line:
    def __init__(self, min, max, num_samples):
        self.window = None
        self.program = None
        self.group = None
        self.vertex_list = None
        self.vertices = None  # interleaved x,y (float32)
        self.min = min
        self.max = max
        self.num_samples = int(num_samples)
        self.buffer = np.zeros(self.num_samples, dtype=np.float32)
        self.current_sample = 0

    def open(self):
        self.window = Window.Window() # Create window
        self.window.open()
        self.program = ShaderProgram(Shader(vs, "vertex"), Shader(fs, "fragment")) # Load Shaders
        self.program["screen_dim"] = (self.window.width, self.window.height) # Set uniforms
        self.group = ShaderGroup(self.program) # Create shader groups

        # Generate interleaved vertex buffer: x across width, y initially mid-line
        x = np.linspace(0, self.window.width - 1, self.num_samples, dtype=np.float32)
        self.vertices = np.empty(2 * self.num_samples, dtype=np.float32)
        self.vertices[0::2] = x
        self.vertices[1::2] = self.buffer
        self.vertex_list = self.program.vertex_list(
            self.num_samples, gl.GL_LINE_STRIP, batch=self.window.batch, group=self.group,
            position=("f", self.vertices)
        )

        @self.window.handle.event
        def on_resize(width, height):
            self.program["screen_dim"] = (width, height)
            self.vertices[0::2] = np.linspace(0, width - 1, self.num_samples, dtype=np.float32)
            self.vertex_list.position[:] = self.vertices
            return pyglet.event.EVENT_HANDLED

    def process_events(self):
        self.window.process_events()

    def plot(self, line_data: np.ndarray):
        data = ((np.asarray(line_data, dtype=np.float32) - self.min) / ((self.max - self.min) / 2.0) - 1.0)
        #print(data)
        new_samples = data.size
        remaining_samples = self.num_samples - self.current_sample
        if new_samples <= remaining_samples:
            self.buffer[self.current_sample:(self.current_sample+new_samples)] = data
            self.current_sample += new_samples
        else:
            self.buffer[self.current_sample:] = data[:remaining_samples]
            self.buffer[:(new_samples - remaining_samples)] = data[remaining_samples:]
            self.current_sample = (new_samples - remaining_samples)
        self.vertices[1::2] = self.buffer
        self.vertex_list.position[:] = self.vertices

    def render(self):
        self.window.render()

    def close(self):
        self.window.close()

# FIN
