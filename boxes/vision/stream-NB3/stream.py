# Stream live camera images
import os, pathlib, time
#import NB3.Vision.camera as Camera
import NB3.Vision.webcam as Camera
import NB3.Server.server as Server

# Specify paths
repo_path = f"{pathlib.Path.home()}/NoBlackBoxes/LastBlackBox"
site_root = f"{repo_path}/boxes/vision/stream-NB3/site"

# Setup Camera
camera = Camera.Camera(width=800, height=600)
camera.start()

# Start Server (for streaming)
interface = Server.get_wifi_interface()
server = Server.Server(root=site_root, interface=interface, autostart=True)

# Stream camera images
try:
    while True:
        frame = camera.capture(mjpeg=True)
        camera.display(frame, server, "camera", jpeg=True)
        time.sleep(0.0333) # (Optional) Limit stream to 30 FPS

except KeyboardInterrupt:
    camera.stop()
    server.stop()
