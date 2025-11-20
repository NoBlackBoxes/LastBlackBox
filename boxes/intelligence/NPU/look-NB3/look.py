# Imports
import time, curses, serial
import numpy as np
import tflite_runtime.interpreter as tflite
import LBB.config as Config
import NB3.Vision.camera as Camera
import NB3.Vision.overlay as Overlay
import NB3.Server.server as Server

# Specify paths
project_path = f"{Config.repo_path}/boxes/intelligence/NPU/look-NB3"

# Specify site root
root = f"{npu_path}/look-NB3/site"

# Open serial port
ser = serial.Serial(port='/dev/ttyUSB0', baudrate = 115200)

# Specify model and labels
model_path = f"{project_path}/look-NB3/model/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite"
#model_path = f"{project_path}/look-NB3/model/movenet_single_pose_thunder_ptq_edgetpu.tflite"

# Load delegate (EdgeTPU)
delegate = tflite.load_delegate('libedgetpu.so.1')

# Create interpreter
interpreter = tflite.Interpreter(model_path=model_path, experimental_delegates=[delegate])
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Open camera, start, and wait for it to settle
camera = Camera.Camera(width=640, height=640, lores_width=320, lores_height=320)
camera.start()
time.sleep(1.0)

# Add Overlay
overlay = Overlay.Overlay()
overlay.timestamp = True
camera.overlay = overlay

# Start Server (for streaming)
interface = Server.get_wifi_interface()
server = Server.Server(root=root, interface=interface)
server.start()

# Initialize interactive terminal
screen = curses.initscr()
curses.noecho()
curses.cbreak()
screen.keypad(True)
screen.nodelay(True)

# Face processing function
def process_face(output_rect):
    ul_x = output_rect[1] * camera.width         # Upper left corner (X)
    ul_y = output_rect[0] * camera.height        # Upper left corner (Y)
    lr_x = output_rect[3] * camera.width         # Lower right corner (X)
    lr_y = output_rect[2] * camera.height        # Lower right corner (Y)
    width = lr_x - ul_x                          # Width
    height = lr_y - ul_y                         # Height
    x = (output_rect[1] + output_rect[3]) / 2.0  # Centroid (X)
    y = (output_rect[0] + output_rect[2]) / 2.0  # Centroid (Y)
    return (ul_x, ul_y, width, height), x, y

# Processing loop
try:
    while True:
        # Check for quit ('q') key
        char = screen.getch()
        if char == ord('q'):
            break

        # Clear screen
        screen.erase()

        # Capture low res image
        frame = camera.capture(lores=True, gray=False)

        # Send to NPU
        interpreter.set_tensor(input_details[0]['index'], np.expand_dims(frame, axis=0))

        # Run inference
        interpreter.invoke()

        # Get output tensors
        output_rects = interpreter.get_tensor(output_details[0]['index'])[0]
        output_scores = interpreter.get_tensor(output_details[2]['index'])[0]
        output_num_faces = interpreter.get_tensor(output_details[3]['index'])[0]

        # Report default status
        screen.addstr(0, 0, f"Status: ...Looking for faces...             ")

        # Count number of detected faces (max = 2)
        face_score_threshold = 0.3
        face_indices = np.where(output_scores > face_score_threshold)[0]        
        num_faces = min(len(face_indices), 2)

        # Respond to faces
        if num_faces == 0:          # If NO face detected
            # ADD YOUR COMMAND RESPONSES AFTER HERE ------->
            ser.write(b'x')                          # Send the Arduino 'x' (the command to stop)
            # <------- ADD YOUR COMMAND BEFORE RESPONSES HERE

            # Report NO faces
            screen.addstr(1, 0, f"-NO-Face Detected              ")
            screen.addstr(2, 0, f"                               ")
            screen.addstr(3, 0, f"                               ")
            overlay.rectangle1 = None
            overlay.rectangle2 = None
        elif num_faces == 1:        # If ONE face detected
            # Process ONE face
            rectangle, x, y = process_face(output_rects[0])

            # ADD YOUR COMMAND RESPONSES AFTER HERE ------->
            ser.write(b'x')                          # Send the Arduino 'x' (the command to stop)
            # <------- ADD YOUR COMMAND BEFORE RESPONSES HERE

            # Report ONE face
            screen.addstr(1, 0, f" -  Face Detected: X: {x:.3f}, Y: {y:.3f} - Score: {output_scores[0]:.2f}")
            screen.addstr(2, 0, f"                               ")
            screen.addstr(3, 0, f"                               ")
            overlay.rectangle1 = rectangle
            overlay.rectangle2 = None
        elif num_faces >= 2:         # If TWO (or more) faces detected
            # Process TWO faces
            rectangle1, x1, y1 = process_face(output_rects[0])
            rectangle2, x2, y2 = process_face(output_rects[1])

            # ADD YOUR COMMAND RESPONSES AFTER HERE ------->
            ser.write(b'x')                          # Send the Arduino 'x' (the command to stop)
            # <------- ADD YOUR COMMAND BEFORE RESPONSES HERE

            # Report TWO faces
            screen.addstr(1, 0, f" -  Face Detected: X: {x1:.3f}, Y: {y1:.3f} - Score: {output_scores[0]:.2f}")
            screen.addstr(2, 0, f" -  Face Detected: X: {x2:.3f}, Y: {y2:.3f} - Score: {output_scores[1]:.2f}")
            screen.addstr(3, 0, f"                               ")
            overlay.rectangle1 = rectangle1
            overlay.rectangle2 = rectangle2
 
        # Add server instructions
        screen.addstr(4, 0, f"🌐 NB3 Server running at http://{server.ip_address}:{server.port}")
        screen.addstr(5, 0, f"    - Press 'q' to Quit")
        screen.addstr(7, 0, f"  ")

        # Update stream
        frame = camera.capture(mjpeg=True)
        server.update_stream("camera", frame)

finally:
    # Shutdown camera
    camera.stop()
    server.stop()

    # Close serial port
    ser.close()

    # Cleanup terminal
    curses.nocbreak()
    screen.keypad(0)
    curses.echo()
    curses.endwin()

#FIN