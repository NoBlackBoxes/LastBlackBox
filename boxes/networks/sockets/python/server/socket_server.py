# Socket Server
#  This program should run on your NB3. It will listen for a connection, and then
#  "serve" data gathered from the serial port to whomever connected.
import socket
import netifaces
import serial

# Configure and open serial port
ser = serial.Serial()
ser.baudrate = 115200
ser.port = '/dev/ttyUSB0'
ser.open()

# Specify IP address and port to use for the socket
ip_address = netifaces.ifaddresses("wlan0")[netifaces.AF_INET][0]['addr'] # Get IP address for WiFi interface
port = 1234

# Create a Socket that listens for connections
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # AF_INET means use IPv4 address and SOCK_STREAM means use TCP (rather than UDP)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # Tell the socket to reuse this address (e.g. if it crashes)
sock.bind((ip_address, port)) # Bind the new socket to the specified IP address and port
sock.listen() # Start listening for connections
print(f"Socket Server listening on {ip_address}:{port}")

# The Socket Server Loop(s)
try:
    while True:                             # This loop will keep checking for a connection
        conn, addr = sock.accept()          # Accept a connection request (waits until one is received)
        print(f"Connected to by {addr}")

        try:
            while True:                     # Stream until the client disconnects
                data = ser.read(16)         # Read 16 samples from the serial port (unsigned bytes) at a time
                conn.sendall(data)          # Send data to socket
                # DEBUG: print(f"Sent {len(data)} bytes to the client")
                # DEBUG: print(f"Bytes Waiting: {ser.in_waiting}")

        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            print("Client disconnected.")
        finally:
            conn.close()
            print("Connection closed; returning to accept new clients.")

except KeyboardInterrupt:
    print("\nShutting down...")

finally:
        sock.close()
        ser.close()

#FIN