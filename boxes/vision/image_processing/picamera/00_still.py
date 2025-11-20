# Capture a still image as a PNG
import time
from picamera2 import Picamera2
import LBB.config as Config

# Specify paths
project_path = f"{Config.repo_path}/boxes/vision/image_processing/picamera"

# Take snapshot
picam2 = Picamera2()
picam2.start()
time.sleep(2)
picam2.capture_file(f"{project_path}/my_still_flip.png")

#FIN