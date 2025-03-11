# Requires installing sshkeyboard: pip install sshkeyboard
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
ser.baudrate = 115200
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
        print("\rQuiting                  ", end="", flush=True)
        ser.write(b'x')
        moving = False
        quit = True
        time.sleep(0.05)
        return False

    # Stopped?
    if not moving:
        # Forward?
        if key == "up":
            print("\rForward (press \'q\') to quit           ", end="", flush=True)
            ser.write(b'f')
            moving = True
            time.sleep(0.05)
        # Backward?
        if key == "down":
            print("\rBackward (press \'q\') to quit           ", end="", flush=True)
            ser.write(b'b')
            moving = True
            time.sleep(0.05)
        # Left?
        if key == "left":
            print("\rLeft (press \'q\') to quit           ", end="", flush=True)
            ser.write(b'l')
            moving = True
            time.sleep(0.05)
        # Right?
        if key == "right":
            print("\rRight (press \'q\') to quit           ", end="", flush=True)
            ser.write(b'r')
            moving = True
            time.sleep(0.05)
    return True
        
def release(key):
    global moving
    if (key == "up") or (key == "down") or (key == "left") or (key == "right"):
        print("\rStopped (press \'q\') to quit           ", end="", flush=True)
        ser.write(b'x')
        moving = False
        time.sleep(0.05)
    return True

# Listen for keyboard events
listen_keyboard(on_press=press, on_release=release, delay_second_char=0.10, until="q")

# Turn on key echo
os.system("stty echo")
print()

# Close serial port
ser.close()

#FIN

