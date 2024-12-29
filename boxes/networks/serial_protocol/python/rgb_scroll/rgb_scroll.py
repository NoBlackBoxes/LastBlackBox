import serial
import time

# Configure serial port
ser = serial.Serial()
ser.baudrate = 115200
ser.port = '/dev/ttyUSB0'

# Open serial port
ser.open()
time.sleep(1.00) # Wait for connection before sending any data

# Send a RGB commands
for i in range(10):
    intensities = list(range(0,255,15)) + list(range(255,0,-15))
    for r in intensities:
        command = f"r:{r:03d}\n"
        ser.write(command.encode())
        #print(command.encode())
        time.sleep(0.01)
    for g in intensities:
        command = f"g:{g:03d}\n"
        #print(command.encode())
        ser.write(command.encode())
        time.sleep(0.01)
    for b in intensities:
        command = f"b:{b:03d}\n"
        #print(command.encode())
        ser.write(command.encode())
        time.sleep(0.01)

# Close serial port
ser.close()

#FIN