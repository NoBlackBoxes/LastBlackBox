# Import libraries
import time, curses, serial, cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import LBB.config as Config
import NB3.Vision.camera as Camera
import NB3.Server.server as Server
from facial_landmarks.facial_processor import FacialLandmarkProcessor, ProcessingConfig

# Specify paths
project_path = f"{Config.repo_path}/course/versions/ai-workshops/02_vision/_resources/python"
site_root = f"{Config.repo_path}/boxes/intelligence/NPU/look-NB3/site"

# Create facial processor
config = ProcessingConfig(enable_face_detection = True,draw_face_boxes = True,draw_landmarks = False,landmark_tracking_confidence = 0.9)

model = FacialLandmarkProcessor(config)

# Load image from camera
camera = Camera.Camera(width=640, height=640, lores_width=320, lores_height=320)
camera.start()
camera.handle.stop_encoder()
time.sleep(1)

# Start network server
interface = Server.get_wifi_interface()
server = Server.Server(root=site_root, interface=interface)
server.start()

# Interactive terminal
screen = curses.initscr()
curses.cbreak()
curses.noecho()
screen.keypad(True)
screen.nodelay(True)

try:
    while True:
        if screen.getch() == ord('q'):
            break
        screen.erase()
        img = camera.capture(gray=False)
        result = model.process_image(img)
        screen.addstr(0, 0, f"NB3 Server running at http://{server.ip_address}:{server.port}")
        screen.addstr(1, 0, f"--------------------------------------------------------------")
        screen.addstr(3, 0, f"Results: ")
        screen.addstr(4, 0, f"Found {len(result.faces)} face[s].")
        if not result.landmarks:
            screen.addstr(5, 0, "No landmarks detected.")
        else:
            screen.addstr(5, 0, f"Found {len(result.landmarks[0].landmarks)} landmarks.")
        screen.addstr(7, 0, f"          - Press 'q' to Quit - ")
        screen.addstr(8, 0 ,f"--------------------------------------------------------------")
        proc = result.processed_image
        #proc = proc.astype(np.uint8)
        #proc = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
        _, proc = cv2.imencode('.jpg', proc, [cv2.IMWRITE_JPEG_QUALITY, 70])
        proc = proc.tobytes()
        #proc = camera.capture(mjpeg=True)
        server.update_stream("camera", proc)
finally:
    camera.stop()
    server.stop()
    curses.nocbreak()
    screen.keypad(0)
    curses.echo()
    curses.endwin()
#
