import os
import time
import NB3.Server.server as Server

# Get user name
username = os.getlogin()

# Specify site root
site_root = f"/home/{username}/NoBlackBoxes/LastBlackBox/boxes/servers/python/NB3_server_example/site"

# Serve
interface = Server.get_wifi_interface()
server = Server.Server(site_root, interface=interface)
server.start()
server.status()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    server.stop()