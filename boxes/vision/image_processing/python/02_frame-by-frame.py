# Measure frame-by-frame difference of each pixel and stream result
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
site_root = f"{Config.repo_path}/boxes/vision/image_processing/python/sites/split"

# Setup Camera
camera = Camera.Camera(width=640, height=480, lores_width=640, lores_height=480)
camera.overlay.timestamp = True
camera.start()

# Start Server (for streaming)
interface = Server.get_wifi_interface()
server = Server.Server(root=site_root, interface=interface, autostart=True)

# Create buffer for previous frame
previous = np.zeros((480,640), dtype=np.uint8)

try:
    while True:
        # Capture RGB frame
        frame = camera.capture(mjpeg=False, lores=False, gray=False)

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Compute absolute difference
        absdiff = cv2.absdiff(gray, previous)
        avg_pixel_diff = np.mean(absdiff)

        # Display raw and processed frame
        camera.overlay.clear()
        camera.overlay.add_label(20, camera.height - 20, f"Average Pixel Change: {avg_pixel_diff:.1f}")
        camera.display(frame, server, "camera", overlay=False, jpeg=False)
        camera.display(absdiff, server, "display", overlay=True, jpeg=False, gray=True)
        
        # Delay
        time.sleep(0.033) # Limit to 30 FPS

        # Store new previous
        previous = gray

except KeyboardInterrupt:
    server.stop()
    camera.stop()
