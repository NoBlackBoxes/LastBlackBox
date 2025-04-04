import pyglet
from pyglet import shapes

# List of all controllers
controllers = pyglet.input.get_controllers()
print(controllers)

# Connect to first (only?) controller
if controllers:
    controller = controllers[0]
    controller.open()

# Initial Controller State
dpad_state = {"up":False, "down":False, "left":False, "right":False}

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
global speed
speed = [0, 0]

# Define shape update function
def update(dt):
    global speed
    circle.x += speed[0]
    circle.y += speed[1]
    if (circle.x > width) or (circle.x < 0):
        speed[0] = -1 * speed[0]
    if (circle.y > height) or (circle.y < 0):
        speed[1] = -1 * speed[1]
    return

# Schedule shape updates (60 Hz)
pyglet.clock.schedule_interval(update, 1.0/60.0)

# Respond to Input
@controller.event
def on_dpad_motion(controller, dpleft, dpright, dpup, dpdown):
    global speed
    global dpad_state
    delta = 0.75
    dx = 0
    dy = 0
    # Up
    if dpup:
        dy = delta
        dpad_state['up'] = True
        print("up (press)")
    else:
        if dpad_state['up']:
            dpad_state['up'] = False
            dy = 0
            print("up (release)")
    # Down
    if dpdown:
        dy = -delta
        dpad_state['down'] = True
        print("down (press)")
    else:
        if dpad_state['down']:
            dpad_state['down'] = False
            dy = 0
            print("down (release)")
    # Left
    if dpleft:
        dx = -delta
        dpad_state['left'] = True
        print("left (press)")
    else:
        if dpad_state['left']:
            dpad_state['left'] = False
            dx=0
            print("left (release)")
    # Right
    if dpright:
        dx = delta
        dpad_state['right'] = True
        print("right (press)")
    else:
        if dpad_state['right']:
            dpad_state['right'] = False
            dx = 0
            print("right (release)")

    # Set ball speed
    speed = [dx, dy]

# Draw Graphics
@window.event
def on_draw():
    window.clear()
    batch.draw()

# Run
pyglet.app.run()

#FIN