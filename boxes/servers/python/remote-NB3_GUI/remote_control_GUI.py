# Imports
import os
import time
import serial
import NB3.Server.server as Server

# Specify site root
username = os.getlogin()
site_root = f"/home/{username}/NoBlackBoxes/LastBlackBox/boxes/servers/python/remote-NB3_GUI/site"

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
   elif command == "do_action":
      # ADD YOUR COMMAND RESPONSES AFTER HERE ------->
      # - What action should your robot do when the "?" is pressed?
      print("Action Triggered!")
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

# Start Server (for streaming)
interface = Server.get_wifi_interface()
server = Server.Server(root=site_root, interface=interface, command_handler=command_handler)
server.start()
server.status()

# Run "NB3 Remote Control Website" Server
try:
    while True:
        time.sleep(1.0)
except KeyboardInterrupt:
    server.stop()
    ser.close()
