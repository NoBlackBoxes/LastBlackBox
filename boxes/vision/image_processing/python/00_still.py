#import NB3.Vision.camera as Camera         # NB3 Camera
import NB3.Vision.webcam as Camera          # Webcam

camera = Camera.Camera(width=1920, height=1080)
camera.start()
camera.save('image.jpg')
camera.stop()
