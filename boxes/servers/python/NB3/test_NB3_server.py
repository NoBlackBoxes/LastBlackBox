import os
import time
import NB3.Server.server as Server

# Get user name
username = os.getlogin()

# Specify site root
root = f"/home/{username}/NoBlackBoxes/LastBlackBox/boxes/servers/python/NB3/site"

# Serve
interface = Server.get_wifi_interface()
server = Server.Server(root, interface=interface)
server.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    server.stop()