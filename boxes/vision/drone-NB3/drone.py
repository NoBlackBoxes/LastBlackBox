# Control your NB3 while streaming the live camera image
import os, pathlib, time
import serial
import numpy as np
import NB3.Sound.speaker as Speaker
import NB3.Vision.camera as Camera
import NB3.Server.server as Server

# Specify paths
repo_path = f"{pathlib.Path.home()}/NoBlackBoxes/LastBlackBox"
site_root = f"{repo_path}/boxes/vision/drone-NB3/site"
sound_path = f"{repo_path}/boxes/vision/drone-NB3/sounds/horn.wav"

# Define command handler
def command_handler(command):
   if command == 'forward':
      ser.write(b'f')
   elif command == 'backward':
      ser.write(b'b')
   elif command == 'left':
      ser.write(b'l')
   elif command == 'right':
      ser.write(b'r')
   elif command == 'stop':
      ser.write(b'x')
   elif command == "play_sound":
      speaker.play_wav(sound_path)
   elif command == "do_action":
      # ADD YOUR COMMAND RESPONSES AFTER HERE ------->
      # - What action should your robot do when the "?" is pressed?
      # <------- ADD YOUR COMMAND BEFORE RESPONSES HERE        
      pass
   else:
      pass

# Configure serial port
ser = serial.Serial()
ser.baudrate = 115200
ser.port = '/dev/ttyUSB0'
ser.open()
time.sleep(1.00)

# Setup speaker
speaker = Speaker.Speaker(1, 2, 'int32', 48000, int(48000 / 10))
speaker.start()

# Setup Camera
camera = Camera.Camera(width=800, height=600)
camera.start()

# Start Server (for streaming)
interface = Server.get_wifi_interface()
server = Server.Server(root=site_root, interface=interface, command_handler=command_handler)
server.start()
server.status()

# Run Drone
try:
    print(f"    - \"Control + C\" to Quit -")
    while True:
        frame = camera.capture(mjpeg=True)
        server.update_stream("camera", frame)
        time.sleep(0.0333) # (Optional) Slow down stream to 30 FPS
except KeyboardInterrupt:
    camera.stop()
    speaker.stop()
    server.stop()
    ser.close()
