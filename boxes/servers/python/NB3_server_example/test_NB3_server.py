import os, pathlib, time
import time
import NB3.Server.server as Server

# Specify site root
repo_path = f"{pathlib.Path.home()}/NoBlackBoxes/LastBlackBox"
site_root = f"{repo_path}/boxes/servers/python/NB3_server_example/site"

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

# FIN