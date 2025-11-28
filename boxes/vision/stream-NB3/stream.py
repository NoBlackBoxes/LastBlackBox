# Stream live camera images from your NB3
import time, cv2
import LBB.config as Config
import importlib.util
if importlib.util.find_spec("picamera2") is not None:   # Is picamera available (only on NB3)?
    import NB3.Vision.camera as Camera                  # NB3 Camera
else:
    import NB3.Vision.webcam as Camera                  # Webcam (PC)
import NB3.Server.server as Server

# Specify site root
site_root = f"{Config.repo_path}/boxes/vision/stream-NB3/site"

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

#FIN