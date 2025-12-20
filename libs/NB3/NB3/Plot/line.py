# -*- coding: utf-8 -*-
"""
NB3 : Plot : Line Class

@author: kampff
"""

# Imports
import numpy as np
import NB3.Plot.axes as Axes
import pyglet
from pyglet import gl
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.graphics import ShaderGroup

# Define shaders
vs = """
#version 330
in vec2 position;
void main(){
    gl_Position = vec4(position, 0.0, 1.0);
}"""

fs = """
#version 330
out vec4 fragColor;
void main(){ fragColor = vec4(1,1,0,0.5);
}"""

# Line Class
class Line:
    def __init__(self, min, max, num_samples, show_cursor=True, show_label=False):
        self.axes = None
        self.program = None
        self.group = None
        self.vertex_list = None
        self.vertices = None  # interleaved x,y (float32)
        self.min = min
        self.max = max
        self.num_samples = int(num_samples)
        self.buffer = np.zeros(self.num_samples, dtype=np.float32)
        self.current_sample = 0
        self.show_cursor = show_cursor
        self.show_label = show_label

    def open(self):
        self.axes = Axes.Axes(show_cursor=self.show_cursor, show_label=self.show_label) # Create Axes
        self.axes.open()
        self.program = ShaderProgram(Shader(vs, "vertex"), Shader(fs, "fragment")) # Load Shaders
        self.group = ShaderGroup(self.program) # Create shader groups

        # Generate interleaved vertex buffer
        x = np.linspace(-1.0, 1.0, self.num_samples, dtype=np.float32)
        self.vertices = np.empty(2 * self.num_samples, dtype=np.float32)
        self.vertices[0::2] = x
        self.vertices[1::2] = self.buffer
        self.vertex_list = self.program.vertex_list(
            self.num_samples, gl.GL_LINE_STRIP, batch=self.axes.batch, group=self.group,
            position=("f", self.vertices)
        )

    def plot(self, line_data: np.ndarray):
        scaled_data = ((np.asarray(line_data, dtype=np.float32) - self.min) / ((self.max - self.min) / 2.0) - 1.0) # -1.0 to 1.0
        new_samples = scaled_data.size
        remaining_samples = self.num_samples - self.current_sample

        if new_samples <= remaining_samples: # Space remaining for new samples in plot buffer
            self.buffer[self.current_sample:(self.current_sample+new_samples)] = scaled_data
            self.current_sample += new_samples
        elif (new_samples >= self.num_samples): # More new samples than entire plot buffer
            self.buffer= scaled_data[(-self.num_samples):]
            self.current_sample = 0
        else: # Wrap around (but not overflow)
            self.buffer[self.current_sample:] = scaled_data[:remaining_samples]
            self.buffer[:(new_samples-remaining_samples)] = scaled_data[remaining_samples:]
            self.current_sample = (new_samples + self.current_sample) % self.num_samples # Wrap-around
        self.vertices[1::2] = self.buffer
        self.vertex_list.position[:] = self.vertices
        if self.show_cursor:
            self.axes.cursor_position = (self.current_sample / self.num_samples)

        # Update plot
        self.axes.process_events()
        self.axes.render()

    def close(self):
        self.axes.close()

#FIN
