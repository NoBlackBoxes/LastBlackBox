# Control your NB3 while streaming the live camera image
import time, cv2, serial
import numpy as np
import LBB.config as Config
import importlib.util
if importlib.util.find_spec("picamera2") is not None:
    import NB3.Vision.camera as Camera                  # NB3 Camera
else:
    import NB3.Vision.webcam as Camera                  # Webcam (PC)
import NB3.Server.server as Server
import NB3.Sound.speaker as Speaker
import NB3.Sound.utilities as Utilities

# Specify paths
site_root = f"{Config.repo_path}/boxes/vision/drone-NB3/site"
sound_path = f"{Config.repo_path}/boxes/vision/drone-NB3/sounds/horn.wav"

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

# Open serial port
ser = serial.Serial(port='/dev/ttyUSB0', baudrate = 115200)
time.sleep(1.00)

# List available sound devices
Utilities.list_devices()

# Get speaker device by name (NB3: "MAX", PC: select based on listed output devices)
output_device = Utilities.get_output_device_by_name("MAX")
if output_device == -1:
    exit("Output device (Audio) not found")

# Setup speaker
speaker = Speaker.Speaker(output_device, 2, 'int32', 48000, int(48000 / 10))
speaker.start()

# Setup Camera
camera = Camera.Camera(width=800, height=600)
camera.start()

# Start Server (for streaming)
interface = Server.get_wifi_interface()
server = Server.Server(root=site_root, interface=interface, command_handler=command_handler, autostart=True)

# Run Drone
try:
    while True:
        frame = camera.capture(mjpeg=True)
        server.update_stream("camera", frame)
        time.sleep(0.0333) # (Optional) Slow down stream to 30 FPS
except KeyboardInterrupt:
    camera.stop()
    speaker.stop()
    server.stop()
    ser.close()

#FIN