# Acquire and save a "still" (single frame) from your NB3 (or PC webcam)
import importlib.util
if importlib.util.find_spec("picamera2") is not None:   # Is picamera available (only on NB3)?
    import NB3.Vision.camera as Camera                  # NB3 Camera
else:
    import NB3.Vision.webcam as Camera                  # Webcam (PC)

# Acquire Still
camera = Camera.Camera(width=1920, height=1080)
camera.start()
camera.save('my_still_image.jpg')
camera.stop()
