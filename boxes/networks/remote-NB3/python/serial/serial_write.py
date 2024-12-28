import serial
import time

# Configure serial port
ser = serial.Serial()
ser.baudrate = 19200
ser.port = '/dev/ttyUSB0'

# Open serial port
ser.open()
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
