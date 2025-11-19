import os, pathlib, time
import cv2
#import NB3.Vision.camera as Camera          # NB3 Camera
import NB3.Vision.webcam as Camera         # Webcam (PC)
import NB3.Server.server as Server

# Specify site root
repo_path = f"{pathlib.Path.home()}/NoBlackBoxes/LastBlackBox"
site_root = f"{repo_path}/boxes/vision/image_processing/python/sites/split"

# Setup Camera
camera = Camera.Camera(width=800, height=600, lores_width=640, lores_height=480)
camera.overlay.timestamp = True
camera.start()

# Set threshold level
threshold_level = 127

# Start Server (for streaming)
interface = Server.get_wifi_interface()
server = Server.Server(root=site_root, interface=interface, autostart=True)

# Processing Loop
try:
    while True:
        # Capture frame
        gray = camera.capture(mjpeg=False, lores=False, gray=True)

        # Apply binary threshold
        _, binary = cv2.threshold(gray, threshold_level, 255, cv2.THRESH_BINARY)

        # Display raw and processed frame
        camera.display(gray, server, "camera", overlay=True, jpeg=False, gray=True)
        camera.display(binary, server, "display", overlay=False, jpeg=False)
        
        # Delay
        time.sleep(0.033) # Limit to 30 FPS

except KeyboardInterrupt:
    server.stop()
    camera.stop()

#FIN