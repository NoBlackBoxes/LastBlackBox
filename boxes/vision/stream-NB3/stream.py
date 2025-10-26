import os
import time
#import NB3.Vision.camera as Camera
import NB3.Vision.webcam as Camera
import NB3.Server.server as Server

# Specify streaming website root
username = os.getlogin()
root = f"/home/{username}/NoBlackBoxes/LastBlackBox/boxes/vision/stream-NB3/site"

# Setup Camera
camera = Camera.Camera(width=800, height=600)
camera.start()

# Start Server (for streaming)
interface = Server.get_wifi_interface()
server = Server.Server(root=root, interface=interface, autostart=True)

# Stream camera images
try:
    while True:
        frame = camera.capture(mjpeg=True)
        server.update_stream("camera", frame)
        time.sleep(0.0333) # (Optional) Limit stream to 30 FPS

except KeyboardInterrupt:
    camera.stop()
    server.stop()
