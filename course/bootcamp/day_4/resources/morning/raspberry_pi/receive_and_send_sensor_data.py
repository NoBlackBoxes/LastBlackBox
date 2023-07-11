import socket
import serial
import struct

# Set up the serial connection
ser = serial.Serial('/dev/ttyUSB0', 9600) # Replace '/dev/ttyUSB0' with the appropriate serial port (you can find it in the Arduino IDE unter 'Tools' --> 'Port:')

def receive_sensor_data():
    # Wait until data has been received
    while True:
        try:
            line = ser.readline().decode().strip()  # Read a line of data from the serial port
            if line:
                value = float(line)
                return value
        except:
            pass


# Raspberry Pi IP address and port number
SERVER_IP = '192.168.137.1'
SERVER_PORT = 5005

# Create a socket and connect to the server
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((SERVER_IP, SERVER_PORT))

def send_to_host_pc(data):
    '''
    Send it via socket to host PC
    '''
    # Send the sensor value over the network
    value_bytes = struct.pack('!f', data)

    client_socket.send(value_bytes)


if __name__ == '__main__':
    try:
        while True:
            data = receive_sensor_data()
            send_to_host_pc(data)
    except KeyboardInterrupt:
        # Close the serial connection and the socket when interrupted
        ser.close()
        client_socket.close()