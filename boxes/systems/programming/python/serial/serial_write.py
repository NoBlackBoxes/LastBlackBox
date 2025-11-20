import serial
import time

# Open serial port
ser = serial.Serial(port='/dev/ttyUSB0', baudrate=115200)
time.sleep(2.00) # Wait for connection before sending any data

# Send a character
ser.write(b'x')
time.sleep(0.05)

# Send a character
ser.write(b'o')
time.sleep(1.00)

# Send a character
ser.write(b'x')
time.sleep(0.05)

# Close serial port
ser.close()
