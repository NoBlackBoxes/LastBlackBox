import os
import time
import cv2
#import NB3.Vision.camera as Camera         # NB3 Camera
import NB3.Vision.webcam as Camera          # Webcam
import NB3.Server.server as Server

# Specify site root
username = os.getlogin()
root = f"/home/{username}/NoBlackBoxes/LastBlackBox/boxes/vision/image_processing/python/sites/slider"

# Setup Camera
camera = Camera.Camera(width=800, height=600, lores_width=640, lores_height=480)
camera.overlay.timestamp = True
camera.start()

# Set threshold level
threshold_level = 127

# Define command handler
def command_handler(command):
   global threshold_level
   if command.startswith('threshold'):
      threshold_level = int(command.split('-')[1])
   else:
      pass

# Start Server (for streaming)
interface = Server.get_wifi_interface()
server = Server.Server(root=root, interface=interface, command_handler=command_handler, autostart=True)

# Processing Loop
try:
    while True:
        # Capture frame
        gray = camera.capture(mjpeg=False, lores=False, gray=True)

        # Apply binary threshold
        _, binary = cv2.threshold(gray, threshold_level, 255, cv2.THRESH_BINARY)

        # Display raw and processed frame
        camera.display(gray, server, "camera", jpeg=False, gray=True)
        camera.display(binary, server, "display", jpeg=False)
        
        # Delay
        time.sleep(0.033) # Limit to 30 FPS

except KeyboardInterrupt:
    server.stop()
    camera.stop()
