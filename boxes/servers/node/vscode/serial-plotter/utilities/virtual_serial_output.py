import serial
import time

# Run this command in a seperate terminal
# > socat -d -d pty,raw,echo=0 pty,raw,echo=0

# Replace with the appropriate device path from the socat output (first device listed)
serial_port = '/dev/pts/6'

# Set up the serial connection
ser = serial.Serial(serial_port, baudrate=9600, timeout=1)

# Send data
for i in range(320):
    value = i % 255 
    message = value.to_bytes(1)
    ser.write(message)
    time.sleep(.01)

# Close the serial connection
ser.close()
