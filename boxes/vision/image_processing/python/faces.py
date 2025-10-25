import os
import time
import cv2
import numpy as np
import NB3.Vision.camera as Camera
#import NB3.Vision.webcam as Camera
import NB3.Vision.overlay as Overlay
import NB3.Server.server as Server

# Get user name
username = os.getlogin()

# Specify site root
root = f"/home/{username}/NoBlackBoxes/LastBlackBox/boxes/vision/image_processing/python/site"

# Setup Camera
camera = Camera.Camera(width=800, height=600, lores_width=640, lores_height=480)
camera.start()

# Add Overlay
overlay = Overlay.Overlay()
overlay.timestamp = True
camera.overlay = overlay

# Start Server (for streaming)
interface = Server.get_wifi_interface()
server = Server.Server(root=root, interface=interface)
server.start()
server.status()

# Load Haar Cascade Detector
det = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Face processing function
def process_face(output_rect):
    ul_x = output_rect[0] * (camera.width/camera.lores_width)   # Upper left corner (X)
    ul_y = output_rect[1] * (camera.height/camera.lores_height) # Upper left corner (Y)
    width = output_rect[2] * (camera.width/camera.lores_width)   # Lower right corner (X)
    height = output_rect[3] * (camera.height/camera.lores_height) # Lower right corner (Y)
    return (ul_x, ul_y, width, height)

try:
    print(f"    - \"Control + C\" to Quit -")
    while True:
        # Capture frame
        gray = camera.capture(lores=True, gray=True)

        faces = det.detectMultiScale(gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(200, 200), # adjust to your image size, maybe smaller, maybe larger?
            flags=cv2.CASCADE_SCALE_IMAGE)
        faces = np.array(faces)
        if faces.shape[0] > 0:
            rectangle1 = process_face(faces[0])
            overlay.rectangle1 = rectangle1
            print(rectangle1)
        else:
            overlay.rectangle1 = None

        # Convert back to RGB so the output remains 3-channel
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        _, encoded = cv2.imencode('.jpg', rgb, [cv2.IMWRITE_JPEG_QUALITY, 70])
        display = encoded.tobytes()

        # Update streams
        frame = camera.mjpeg()
        server.update_stream("camera", frame)
        server.update_stream("display", display)
        time.sleep(0.0333) # (Optional) Slow down stream to 30 FPS

except KeyboardInterrupt:
    server.stop()
    camera.stop()
