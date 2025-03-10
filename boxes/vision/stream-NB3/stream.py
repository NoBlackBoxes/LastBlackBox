import os
import time
import NB3.Vision.camera as Camera
import NB3.Server.server as Server

# Get user name
username = os.getlogin()

# Specify site root
root = f"/home/{username}/NoBlackBoxes/LastBlackBox/boxes/vision/stream-NB3/site"

# Setup Camera
camera = Camera.Camera(width=1280, height=720)
camera.start()

# Start Server (for streaming)
interface = Server.get_wifi_interface()
server = Server.Server(root=root, interface=interface)
server.start()
server.status()

# Stream camera images
try:
    print(f"    - \"Control + C\" to Quit -")
    while True:
        frame = camera.mjpeg()
        server.update_stream("camera", frame)
        time.sleep(0.0333) # (Optional) Slow down stream to 30 FPS

except KeyboardInterrupt:
    camera.stop()
    server.stop()
