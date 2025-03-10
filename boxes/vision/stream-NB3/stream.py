import os
import time
import NB3.Vision.camera as Camera
import NB3.Server.server as Server

# Get user name
username = os.getlogin()

# Specify site root
root = f"/home/{username}/NoBlackBoxes/LastBlackBox/boxes/vision/stream-NB3"

# Setup Camera
camera = Camera.Camera(width=1280, height=720)
camera.start()

# Start Server (for streaming)
interface = Server.get_wifi_interface()
server = Server.Server(root=html_path, interface=interface)
server.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    camera.stop()
    server.stop()
