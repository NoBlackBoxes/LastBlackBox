# Import libraries
import os
import time
import curses
import serial
import numpy as np
import tflite_runtime.interpreter as tflite

# Import modules
import NB3.Vision.camera as Camera

# Get user name
username = os.getlogin()

# Set base path
npu_path = f"/home/{username}/NoBlackBoxes/LastBlackBox/boxes/intelligence/NPU"

# Configure serial port
ser = serial.Serial()
ser.baudrate = 115200
ser.port = '/dev/ttyUSB0'
ser.open()

# Specify model and labels
model_path = f"{npu_path}/look-NB3/model/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite"
#model_path = f"{npu_path}/look-NB3/model/movenet_single_pose_thunder_ptq_edgetpu.tflite"

# Load delegate (EdgeTPU)
delegate = tflite.load_delegate('libedgetpu.so.1')

# Create interpreter
interpreter = tflite.Interpreter(model_path=model_path, experimental_delegates=[delegate])
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Open camera, start, and wait for it to settle
camera = Camera.Camera(type='picamera2', device=0, width=320, height=320, format='RGB')
camera.start()
time.sleep(1.0)

# Initialize interactive terminal
screen = curses.initscr()
curses.noecho()
curses.cbreak()
screen.keypad(True)
screen.nodelay(True)

# Processing loop
try:
    while True:
        # Check for quit ('q') key
        char = screen.getch()
        if char == ord('q'):
            break

        # Clear screen
        screen.erase()

        # Capture latest image
        frame = camera.latest()

        # Send to NPU
        interpreter.set_tensor(input_details[0]['index'], np.expand_dims(frame, axis=0))

        # Run inference
        interpreter.invoke()

        # Get output tensors
        output_rects = interpreter.get_tensor(output_details[0]['index'])[0]
        output_scores = interpreter.get_tensor(output_details[2]['index'])[0]
        output_num_faces = interpreter.get_tensor(output_details[3]['index'])[0]

        # Process best detected face
        screen.addstr(0, 0, 'Status: ...Looking...')       
        if output_scores[0] > 0.1:
            face_rect_x = output_rects[0][1] * camera.width
            face_rect_y = output_rects[0][0] * camera.height
            x2 = output_rects[0][3] * camera.width
            y2 = output_rects[0][2] * camera.height
            face_rect_width = x2 - face_rect_x
            face_rect_height = y2 - face_rect_y
            x_mid = (face_rect_x + face_rect_width) / 2.0
            y_mid = (face_rect_y + face_rect_height) / 2.0
            screen.addstr(1, 0, f" -  Face Detected: X: {x_mid:.1f}, Y: {y_mid:.1f}")
        else:
            screen.addstr(1, 0, f"-NO-Face Detected")

finally:
    # Shutdown camera
    camera.stop()

    # Close serial port
    ser.close()

    # Cleanup terminal
    curses.nocbreak()
    screen.keypad(0)
    curses.echo()
    curses.endwin()

#FIN