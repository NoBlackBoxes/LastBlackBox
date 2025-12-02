# Acquire and save a "still" (single frame) from your NB3 (or PC webcam)
import LBB.config as Config
import importlib.util
if importlib.util.find_spec("picamera2") is not None:
    import NB3.Vision.camera as Camera                  # NB3 Camera
else:
    import NB3.Vision.webcam as Camera                  # Webcam (PC)

# Specify paths
project_path = f"{Config.repo_path}/boxes/vision/image_processing/python/offline/"

# Acquire Still
camera = Camera.Camera(width=1920, height=1080)
camera.start()
camera.save(f"{project_path}/my_00_still.jpg")
camera.stop()

#FIN