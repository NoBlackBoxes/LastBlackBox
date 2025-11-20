# Imports
import time, serial
import LBB.config as Config
import NB3.Server.server as Server

# Specify site root
site_root = f"{Config.repo_path}/boxes/servers/python/remote-NB3_GUI/site"

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

# Open serial port
ser = serial.Serial(port='/dev/ttyUSB0', baudrate=115200)
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
