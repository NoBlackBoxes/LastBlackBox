# Socket Client (pyglet)
#  This program should run on your PC. It will connect to your NB3 and then receive data streamed
# over the socket and plot it in a real-time graph, with LOW latency, using OpenGL.
import socket
import numpy as np
import pyglet
from pyglet import gl
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.graphics import ShaderGroup

# Specify host IP address and port to use for the socket.
ip_address = '192.168.1.80'  # Use the IP of your NB3 (the server)
port = 1234  # Use the same port as specified in the socket_server

# Create a Socket that establish the server connection
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(60)  # Timeout after 60 seconds of inactivity
sock.connect((ip_address, port))
print(f"Connected to server at {ip_address}:{port}")

# Setup plotting
num_samples_to_plot = 1024
N = num_samples_to_plot
plot_buffer = 127*np.ones(num_samples_to_plot, dtype=np.uint8)
write_pos = 0

# Setup Window
W, H = 900, 300

# ----- window -----
win = pyglet.window.Window(W, H, caption="pyglet 2.x line plot (uint8)")
gl.glEnable(gl.GL_BLEND)
gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
gl.glLineWidth(1.0)

# Simple shader: screen-space transform (position is in pixel coords)
vs = """
#version 330
uniform vec2 u_screen;        // (W, H)
in vec2 position;
void main() {
    vec2 ndc = (position / u_screen) * 2.0 - 1.0;
    ndc.y = -ndc.y;            // flip Y so 0 is bottom
    gl_Position = vec4(ndc, 0.0, 1.0);
}
"""
fs = """
#version 330
out vec4 fragColor;
void main() { fragColor = vec4(1.0,1.0,1.0,1.0); }   // white
"""
program = ShaderProgram(Shader(vs, 'vertex'), Shader(fs, 'fragment'))
program['u_screen'] = (W, H)
group = ShaderGroup(program)   # <- ensure the shader is bound when the batch draws

# Build interleaved x,y vertex data (float32). Start with zeros for y.
x = np.linspace(0, W-1, N, dtype=np.float32)
y = np.zeros(N, dtype=np.float32)
verts = np.empty(2*N, dtype=np.float32)
verts[0::2] = x
verts[1::2] = y

# Create the vertex list ONCE. (GL_LINE_STRIP)
batch = pyglet.graphics.Batch()
vlist = program.vertex_list(
    N, gl.GL_LINE_STRIP, batch=batch,
    position=('f', verts)   # attribute name must match the shader: "position"
)

def update_plot(buf_uint8: np.ndarray):
    # map 0..255 -> 0..H-1, and update only Ys in the underlying array
    verts[1::2] = buf_uint8.astype(np.float32) * (H-1) / 255.0
    vlist.position[:] = verts  # push updated data (fast path)

@win.event
def on_draw():
    win.clear()
    batch.draw()

@win.event
def on_resize(w, h):
    program['u_screen'] = (w, h)
    # keep X spanning the new width
    verts[0::2] = np.linspace(0, w-1, N, dtype=np.float32)
    vlist.position[:] = verts

# The Socket Client Loop
try:
    while True: # Receive 16 bytes of data
        bytes = sock.recv(16)
        if not bytes:
            print("No data received, waiting...")  # Log when no data is received
            continue  # Keep the connection alive
        data = np.frombuffer(bytes, dtype=np.uint8)
        n = len(data)

        # Write data into circular buffer
        end = write_pos + n
        if end <= num_samples_to_plot:
            plot_buffer[write_pos:end] = data
        else:
            k = num_samples_to_plot - write_pos
            plot_buffer[write_pos:] = data[:k]
            plot_buffer[:end - num_samples_to_plot] = data[k:]
        write_pos = (write_pos + n) % num_samples_to_plot

        # Plot data
        update_plot(plot_buffer)
        #DEBUG: print(f"Received: {data}")
        print(f"Received: {data}")

except socket.timeout:
    print("Socket timed out. No data received for 60 seconds.")
except Exception as e:
    print(f"Error: {e}")
finally:
    sock.close()

# FIN



# plotting class
    

# pip install pyglet
import numpy as np
import pyglet
from pyglet import gl
from pyglet.graphics.shader import Shader, ShaderProgram, ShaderGroup

class PygletLinePlot:
    def __init__(self):
        self.win = None
        self.program = None
        self.group = None
        self.batch = None
        self.vlist = None
        self.W = self.H = self.N = 0
        self.verts = None  # interleaved x,y float32

    def open(self, n_samples=1024, width=900, height=300, title="pyglet plot"):
        """Create window + GL objects. Call from the main thread."""
        self.N, self.W, self.H = int(n_samples), int(width), int(height)

        self.win = pyglet.window.Window(self.W, self.H, caption=title, vsync=True)
        self.win.switch_to()  # ensure GL context is current
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glLineWidth(1.0)

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
        void main(){ fragColor = vec4(1,1,1,1); }"""

        self.program = ShaderProgram(Shader(vs, "vertex"), Shader(fs, "fragment"))
        self.program["u_screen"] = (self.W, self.H)
        self.group = ShaderGroup(self.program)

        # Interleaved vertex buffer: x across width, y initially mid-line
        x = np.linspace(0, self.W - 1, self.N, dtype=np.float32)
        self.verts = np.empty(2 * self.N, dtype=np.float32)
        self.verts[0::2] = x
        self.verts[1::2] = (self.H - 1) * 0.5  # mid

        self.batch = pyglet.graphics.Batch()
        self.vlist = self.program.vertex_list(
            self.N, gl.GL_LINE_STRIP, batch=self.batch, group=self.group,
            position=("f", self.verts)
        )

        @self.win.event
        def on_resize(w, h):
            self.W, self.H = w, h
            self.program["u_screen"] = (w, h)
            # keep x spanning new width
            self.verts[0::2] = np.linspace(0, w - 1, self.N, dtype=np.float32)
            self.vlist.position[:] = self.verts
            return pyglet.event.EVENT_HANDLED

    def process_events(self):
        """Pump window events so the OS doesn't mark the window unresponsive."""
        if self.win:
            self.win.dispatch_events()

    def draw_data(self, buf_uint8: np.ndarray):
        """Update Y values from a uint8 array (0..255)."""
        y = np.asarray(buf_uint8, dtype=np.uint8)
        if y.size != self.N:
            # If sizes differ, take the tail or pad on the left with last value.
            if y.size > self.N:
                y = y[-self.N:]
            else:
                pad = np.full(self.N - y.size, y[-1] if y.size else 0, np.uint8)
                y = np.concatenate([pad, y])
        # map to pixel Y and push to GPU (update only Ys = odd indices)
        self.verts[1::2] = y.astype(np.float32) * (self.H - 1) / 255.0
        self.vlist.position[:] = self.verts

    def render(self):
        """Draw the current frame NOW (no scheduling)."""
        if self.win:
            self.win.clear()
            self.batch.draw()
            self.win.flip()

    def close(self):
        if self.win:
            self.win.close()
            self.win = None




# User


plot = PygletLinePlot()
plot.open(n_samples=1024)

# your blocking loop (e.g., socket recv in same thread)
while True:
    # ... recv() or produce a new uint8 buffer of length 1024 ...
    buf = latest_uint8_array  # 0..255
    plot.draw_data(buf)       # push data
    plot.process_events()     # handle window events
    plot.render()             # draw immediately

