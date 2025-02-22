import NB3.Vision.camera as Camera

camera = Camera.Camera(type='picamera2', device=0, width=640, height=480, format='BGR')
camera.start()
frame = camera.latest()
camera.save('image.jpg')
camera.stop()
