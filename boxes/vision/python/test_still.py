import NB3.Vision.camera as Camera

camera = Camera.Camera(type='picamera2', device=0, width=320, height=320, format='RGB')
camera.start()
frame = camera.latest()
camera.save('image.jpg')
camera.stop()
