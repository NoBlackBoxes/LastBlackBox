# Capture a "vertically flipped" still image as a PNG
import time
from picamera2 import Picamera2
import libcamera
import LBB.config as Config

# Specify paths
project_path = f"{Config.repo_path}/boxes/vision/image_processing/picamera"

# Init camera
picam2 = Picamera2()

# Configure camera transform
camera_config = picam2.create_preview_configuration()
camera_config["transform"] = libcamera.Transform(hflip=0, vflip=1)
picam2.configure(camera_config)

# Take snapshot
picam2.start()
time.sleep(2)
picam2.capture_file(f"{project_path}/my_still_flip.png")

#FIN