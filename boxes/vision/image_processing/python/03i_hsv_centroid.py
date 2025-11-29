# Color threshold a Huw, Sat, and Value image to isolate a specific color
# - Find the largest binary "blob" of isolated pixels
# - Include interactive sliders on the streaming page to adjust HSV threshold levels
import time, cv2
import numpy as np
import LBB.config as Config
import importlib.util
if importlib.util.find_spec("picamera2") is not None:
    import NB3.Vision.camera as Camera                  # NB3 Camera
else:
    import NB3.Vision.webcam as Camera                  # Webcam (PC)
import NB3.Server.server as Server

# Specify site root
site_root = f"{Config.repo_path}/boxes/vision/image_processing/python/sites/sliders"

# Setup Camera
camera = Camera.Camera(width=800, height=600, lores_width=640, lores_height=480)
camera.overlay.timestamp = True
camera.overlay.timestamp_position = (20, camera.height - 40)
camera.start()

# Define HSV range for green color
hue_level = 60
sat_level = 90
val_level = 70

# Define command handler
def command_handler(command):
   global hue_level
   global sat_level
   global val_level
   if command.startswith('hue'):
      hue_level = int(command.split('-')[1])
   if command.startswith('sat'):
      sat_level = int(command.split('-')[1])
   if command.startswith('val'):
      val_level = int(command.split('-')[1])
   else:
      pass

# Start Server (for streaming)
interface = Server.get_wifi_interface()
server = Server.Server(root=site_root, interface=interface, command_handler=command_handler, autostart=True)

try:
    while True:
        # Capture frame (RGB)
        rgb = camera.capture(mjpeg=False, lores=False, gray=False)

        # Convert to HSV
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

        # Threshold for HSV
        hue_min = max(0, hue_level-25)
        hue_max = min(255, hue_level+25)
        lower = np.array([hue_min, sat_level, val_level])
        upper = np.array([hue_max, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        camera.overlay.clear()
        if contours:
            # Find the largest contour based on area
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            camera.overlay.add_rectangle(x, y, w, h)
            camera.overlay.add_label(x, y, f"X: {x:.1f}, Y: {y:.1f}")

        # Display raw and processed frames
        camera.display(rgb, server, "camera", overlay=True, jpeg=False)
        camera.display(mask, server, "display", overlay=True, jpeg=False, gray=True)

        # Delay
        time.sleep(0.067) # Limit to 15 FPS

except KeyboardInterrupt:
    server.stop()
    camera.stop()
