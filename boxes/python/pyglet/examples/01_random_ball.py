import pyglet
from pyglet import shapes
import random

# Create (yellow) Window
width = 256
height = 256
window = pyglet.window.Window(width, height)
pyglet.gl.glClearColor(0.75,0.75,0.0,1.0)

# Create shape (circle)
batch = pyglet.graphics.Batch()
x = width//2
y = height//2
circle = shapes.Circle(x, y, 11, color=(0, 0, 0), batch=batch)
speed = [random.randint(-5,5), random.randint(-5,5)]

# Define shape update function
def update(dt):
    circle.x += speed[0]
    circle.y += speed[1]
    if (circle.x > width) or (circle.x < 0):
        speed[0] = -1 * speed[0]
    if (circle.y > height) or (circle.y < 0):
        speed[1] = -1 * speed[1]
    return

# Schedule shape updates (60 Hz)
pyglet.clock.schedule_interval(update, 1.0/60.0)

# Draw
@window.event
def on_draw():
    window.clear()
    batch.draw()

# Run
pyglet.app.run()

#FIN