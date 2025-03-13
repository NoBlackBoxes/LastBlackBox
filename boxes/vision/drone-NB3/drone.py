# Imports
import os
import io
import cv2
import time
import serial
import socket
import threading
import netifaces
import numpy as np
import NB3.Sound.speaker as Speaker
import NB3.Vision.camera as Camera
import NB3.Server.server as Server

# Get user name
username = os.getlogin()

# Specify site root
root = f"/home/{username}/NoBlackBoxes/LastBlackBox/boxes/vision/drone-NB3/site"
sound_path = f"/home/{username}/NoBlackBoxes/LastBlackBox/boxes/vision/drone-NB3/sounds/horn.wav"

# Define sound effect player
def play_sound(wav_path):
    speaker.play_wav(sound_path)
    pass

# Define command handler
def command_handler(command):
    if command == 'forward':
       ser.write(b'f')
    if command == 'backward':
       ser.write(b'b')
    if command == 'left':
       ser.write(b'l')
    if command == 'right':
       ser.write(b'r')
    if command == 'stop':
       ser.write(b'x')
    elif command == "play_sound":
        play_sound(sound_path)
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
server = Server.Server(root=root, interface=interface, command_handler=command_handler)
server.start()
server.status()

# Run Drone
try:
    print(f"    - \"Control + C\" to Quit -")
    while True:
        frame = camera.mjpeg()
        server.update_stream("camera", frame)
        time.sleep(0.0333) # (Optional) Slow down stream to 30 FPS
except KeyboardInterrupt:
    camera.stop()
    speaker.stop()
    server.stop()
    ser.close()
