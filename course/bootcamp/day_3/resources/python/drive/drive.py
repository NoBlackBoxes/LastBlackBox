# Requires installing sshkeyboard: pip3 install sshkeyboard
import os
import serial
import time
from sshkeyboard import listen_keyboard

# Turn off key echo
os.system("stty -echo")

# Specify global states
quit = False
moving = False

# Configure serial port
ser = serial.Serial()
ser.baudrate = 19200
ser.port = '/dev/ttyUSB0'

# Open serial port
ser.open()
time.sleep(2.00) # Wait for connection before sending any data

# Start key listening thread
def press(key):
    global quit
    global moving

    # Quit?
    if key == "q":
        print("Quit")
        ser.write(b'x')
        moving = False
        quit = True
        time.sleep(0.05)
        return False

    # Stopped?
    if not moving:
        # Forward?
        if key == "up":
            print("Forward")
            ser.write(b'f')
            moving = True
            time.sleep(0.05)
        # Backward?
        if key == "down":
            print("Backward")
            ser.write(b'b')
            moving = True
            time.sleep(0.05)
        # Left?
        if key == "left":
            print("Left")
            ser.write(b'l')
            moving = True
            time.sleep(0.05)
        # Right?
        if key == "right":
            print("Right")
            ser.write(b'r')
            moving = True
            time.sleep(0.05)
    return True
        
def release(key):
    global moving
    if (key == "up") or (key == "down") or (key == "left") or (key == "right"):
        print("Stop")
        ser.write(b'x')
        moving = False
        time.sleep(0.05)
    return True

# Listen for keyboard events
listen_keyboard(on_press=press, on_release=release, delay_second_char=0.10, until="q")

# Turn on key echo
os.system("stty echo")

# Close serial port
ser.close()

#FIN

