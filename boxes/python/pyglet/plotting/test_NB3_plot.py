import time
import numpy as np
import NB3.Plot.line as Line

# Open line plot
line = Line.Line(min=50, max=210, num_samples=600)
line.open()

# Plotting loop
while True:
    try:
        # Generate random data
        data = np.random.randint(120, 140, 16)
        line.plot(data)           # push data
        time.sleep(0.02)
    except KeyboardInterrupt:
        break

# Cleanup
line.close()

#FIN