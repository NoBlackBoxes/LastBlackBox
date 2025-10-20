import time
import numpy as np
import NB3.Plot.line as Line

# Open line plot
line = Line.Line(512, 256, 512, 0, num_samples=255)
line.open()

# your blocking loop (e.g., socket recv in same thread)
while True:
    try:
        # Generate random data
        data = np.random.randint(120, 135, 512)
        line.draw_data(data)      # push data
        line.process_events()     # handle window events
        line.render()             # draw immediately
        time.sleep(0.02)
    except KeyboardInterrupt:
        break

# Cleanup
line.close()

# FIN