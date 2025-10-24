import time
import numpy as np
import NB3.Plot.window as Window

# Open line plot
window = Window.Window()
window.open()

# your blocking loop (e.g., socket recv in same thread)
while True:
    try:
        time.sleep(0.02)
    except KeyboardInterrupt:
        break

# Cleanup
window.close()

# FIN