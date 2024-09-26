# Capture a PNG
import time
from picamera2 import Picamera2

# Get user name
import os
username = os.getlogin()

# Set Repo Root
LBB_ROOT = '/home/' + username + '/NoBlackBoxes/LastBlackBox'

# Take snapshot
picam2 = Picamera2()
picam2.start()
time.sleep(2)
picam2.capture_file(LBB_ROOT + "/_tmp/images/tests/test.png")

#FIN