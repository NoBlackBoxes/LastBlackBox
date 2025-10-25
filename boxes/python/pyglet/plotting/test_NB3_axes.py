import time
import NB3.Plot.axes as Axes

# Open plot axes
axes = Axes.Axes()
axes.open()

# your blocking loop (e.g., socket recv in same thread)
while True:
    try:
        time.sleep(0.02)
    except KeyboardInterrupt:
        break

# Cleanup
axes.close()

# FIN