import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
millis = lambda: int(round(time.time() * 1000))
times = []
for i in range(int(50e3)):
    start = millis()
    time.sleep(.001)
    stop = millis()
    times.append(stop - start)
plt.hist(times, [4, 5, 6, 7, 8, 9, 10])
plt.show()
