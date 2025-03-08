import NB3.Vision.camera as Camera

camera = Camera.Camera(width=1920, height=1080)
camera.start()
frame = camera.latest()
camera.save('image.jpg')
camera.stop()
