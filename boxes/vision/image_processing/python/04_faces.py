import os, pathlib, time
import cv2
import numpy as np
#import NB3.Vision.camera as Camera         # NB3 Camera
import NB3.Vision.webcam as Camera        # Webcam (PC)
import NB3.Server.server as Server

# Specify site root
repo_path = f"{pathlib.Path.home()}/NoBlackBoxes/LastBlackBox"
site_root = f"{repo_path}/boxes/vision/image_processing/python/sites/single"

# Setup Camera
camera = Camera.Camera(width=640, height=480, lores_width=640, lores_height=480)
camera.start()

# Load Haar Cascade Detector
det = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Start Server (for streaming)
interface = Server.get_wifi_interface()
server = Server.Server(root=site_root, interface=interface, autostart=True)

# Face processing function
def process_face(output_rect):
    ul_x = output_rect[0] * (camera.width/camera.lores_width)   # Upper left corner (X)
    ul_y = output_rect[1] * (camera.height/camera.lores_height) # Upper left corner (Y)
    width = output_rect[2] * (camera.width/camera.lores_width)   # Lower right corner (X)
    height = output_rect[3] * (camera.height/camera.lores_height) # Lower right corner (Y)
    return (ul_x, ul_y, width, height)

# Processing Loop
try:
    while True:
        # Capture frame
        gray = camera.capture(mjpeg=False, lores=False, gray=True)

        # Detect faces
        faces = det.detectMultiScale(gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(200, 200), # adjust to your image size, maybe smaller, maybe larger?
            flags=cv2.CASCADE_SCALE_IMAGE)

        # Process Faces
        camera.overlay.clear()
        for face in faces:
            if face.shape[0] > 0:
                x, y, w, h = process_face(face)
                camera.overlay.add_rectangle(x, y, w, h)

        # Display raw and processed frame
        camera.display(gray, server, "display", overlay=True, jpeg=False, gray=True)
        
        # Delay
        time.sleep(0.033) # Limit to 30 FPS

except KeyboardInterrupt:
    server.stop()
    camera.stop()
