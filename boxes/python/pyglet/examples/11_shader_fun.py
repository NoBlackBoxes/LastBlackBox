import pyglet
from pyglet.graphics.shader import Shader, ShaderProgram

# Prepare Window
width = 256
height = 256
window = pyglet.window.Window(width, height)
pyglet.gl.glClearColor(0.0,0.0,0.0,1.0) # RGBA

# Report OpenGL Vendor and Version
vendor = pyglet.gl.gl_info.get_vendor()
version = pyglet.gl.gl_info.get_version()
print(f"Open GL Version (Vendor): {version} ({vendor})")

# Shader Code
vertex_source = """#version 150 core
    in vec4 position;
    void main()
    {
        gl_Position = position;
    }
"""

fragment_source = """#version 150 core
    out vec4 final_color;
    void main()
    {
        final_color.r = mod(gl_FragCoord.x, 64) / 60;
        final_color.g = mod(gl_FragCoord.y, 64) / 60;
        final_color.b = 0.0;
        final_color.a = 1.0;
    }
"""

# Compile and Load Shader
vert_shader = Shader(vertex_source, 'vertex')
frag_shader = Shader(fragment_source, 'fragment')
program = ShaderProgram(vert_shader, frag_shader)
#for attribute in program.attributes.items():
#    print(attribute)
#
#for uniform in program.uniforms.items():
#    print(uniform)

# Load Geometry (Vertices)
batch = pyglet.graphics.Batch()
vlist = program.vertex_list(6, pyglet.gl.GL_TRIANGLES, batch=batch)
vlist.position[:] = [-1,1,0,1, 1,-1,0,1, 1,1,0,1,  -1,1,0,1, 1,-1,0,1, -1,-1,0,1]

# Event Handlers
@window.event
def on_draw():
    window.clear()
    program.use()
    batch.draw()
    program.stop()

# Run Application
pyglet.app.run()

#FIN
