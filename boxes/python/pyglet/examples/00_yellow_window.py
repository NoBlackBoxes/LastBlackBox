import pyglet

width = 256
height = 256
window = pyglet.window.Window(width, height)
pyglet.gl.glClearColor(0.75,0.75,0.0,1.0) # RGBA

@window.event
def on_draw():
    window.clear()

pyglet.app.run()
#FIN