import pyglet

# List all controllers
controllers = pyglet.input.get_controllers()
print(controllers)

# Connect to first (only?) controller
if controllers:
    controller = controllers[0]
    controller.open()

# Initial Controller State
dpad_state = {"up":False, "down":False, "left":False, "right":False}

# Respond to Input
@controller.event
def on_dpad_motion(controller, dpleft, dpright, dpup, dpdown):
    global dpad_state
    # Up
    if dpup:
        dpad_state['up'] = True
        print("up (press)")
    else:
        if dpad_state['up']:
            dpad_state['up'] = False
            print("up (release)")
    # Down
    if dpdown:
        dpad_state['down'] = True
        print("down (press)")
    else:
        if dpad_state['down']:
            dpad_state['down'] = False
            print("down (release)")
    # Left
    if dpleft:
        dpad_state['left'] = True
        print("left (press)")
    else:
        if dpad_state['left']:
            dpad_state['left'] = False
            print("left (release)")
    # Right
    if dpright:
        dpad_state['right'] = True
        print("right (press)")
    else:
        if dpad_state['right']:
            dpad_state['right'] = False
            print("right (release)")

# Run Application
pyglet.app.run()

#FIN