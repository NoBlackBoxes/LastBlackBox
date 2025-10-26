# Impots
import os
import cv2
import time
import psutil
import numpy as np
import NB3.Vision.camera as Camera
import NB3.Server.server as Server

# Get user name
username = os.getlogin()

# Specify site root
root = f"/home/{username}/NoBlackBoxes/LastBlackBox/boxes/vision/stream-NB3/site"

# Setup Camera
camera = Camera.Camera(width=800, height=600)
camera.start()

# Start Server (for streaming)
interface = Server.get_wifi_interface()
server = Server.Server(root=root, interface=interface)
server.start()
server.status()

# Benchmarking variables
alpha = 0.01  # Smoothing factor for running average (0 < alpha < 1)
avg_cpu = 0.0
avg_bitrate = 0.0
start_time = time.time()
count = 0

try:
    print(f"    - \"Control + C\" to Quit -")
    while True:
        # Measuring timing
        elapsed_time = time.time() - start_time 
        start_time = time.time()

        # Update streams
        frame = camera.capture(mjpeg=True)
        frame_size = len(frame)
        server.update_stream("camera", frame)
        time.sleep(0.0333) # (Optional) Slow down stream to 30 FPS

        # Update benchmarks
        cpu_usage = psutil.cpu_percent(interval=None)           # Get current CPU usage
        bitrate = (frame_size * 8) / elapsed_time
        if count > 2:
            avg_cpu = alpha * cpu_usage + (1 - alpha) * avg_cpu
            avg_bitrate = alpha * bitrate + (1 - alpha) * avg_bitrate
            print(f"\rCPU: {avg_cpu:.2f}% | Bitrate: {avg_bitrate / 1e6:.2f} Mbps", end="", flush=True)
        else:
            avg_cpu = cpu_usage
            avg_bitrate = bitrate
        count = count + 1

except KeyboardInterrupt:
    print()
    camera.stop()
    server.stop()
