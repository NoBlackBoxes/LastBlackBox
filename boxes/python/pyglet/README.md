# Python : Pyglet

Pyglet is a Python OpenGL framework. It allows full access to graphics hardware on your computer via Python. It also has many other convenient features for getting inputs from keyboards/mice/joysticks and playing sounds. It should work on Windows, Mac, and Linux.

## Installation
- Activate your virtual environment

```bash
pip install pyglet
```

## Geting Started

Let's open a window and make it's background yellow. [Yellow Window](examples/00_yellow_window.py)

```python
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
```

## A Randomly Bouncing Ball

Let's draw a shape and make it move in a random direction...until it hits the edge of the screen. [Random Ball](examples/01_random_ball.py)

