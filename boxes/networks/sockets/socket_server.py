# Socket Server
#  This program should run on your NB3. It will listen for a connection, and (when connected)
#  will "serve" data gathered from the serial port connected to the Arduino to whomever connected.
import socket
import serial
import time

# Configure serial port
ser = serial.Serial()
ser.baudrate = 115200
ser.port = '/dev/ttyUSB0'

# Open serial port
ser.open()
time.sleep(2.00) # Wait for connection

# Specify host IP address and port to use for the socket. 
#  Note: using an empty IP "host" address will tell the program to select whatever address is available,
#  which will most likely be the address assigned to your WiFi device. This is just a convenience.
host = ''
port = 1234

# Create a Socket that listens for connections
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # AF_INET means use IPv4 address and SOCK_STREAM means use TCP (rather than UDP)
s.bind((host, port)) # Bind the new socket to the specified host (address) and port
s.listen() # Start listening
print(f"Server listening on {host}:{port}")

# This loop will keep checking for a connection
while True:
    conn, addr = s.accept() # Accept a connection request
    print(f"Connected by {addr}")
    
    # Read data from serial port
    byte = ser.read()
    time.sleep(0.05)

    # Send random data to the client
    for _ in range(5):  # Send 5 random buffers of data
        buffer_size = random.randint(10, 100)  # Random buffer size between 10 and 100 bytes
        data = generate_random_data(buffer_size)
        conn.sendall(data)
        print(f"Sent {len(data)} bytes to the client")
        time.sleep(1)  # Simulate some delay between sending data
    
    print("Closing connection with the client.")
    conn.close()

# FIN
