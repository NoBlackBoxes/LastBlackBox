# Simple HTTP server
import os
import socket

# Load home page data (index.html)
path = 'index.html'
file = open(path,"r")
html = file.read()
file.close()

# Set host (this computer) and port (1234)
HOST, PORT = '', 1234

# Open socket
listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listen_socket.bind((HOST, PORT))
listen_socket.listen(1)
print(f'Serving HTTP on port {PORT} ...')

# Serve incoming connections
while True:
    # Listen for a connection
    client_connection, client_address = listen_socket.accept()

    # When a connection arrives, retrieve/decode HTTP request
    request_data = client_connection.recv(1024)
    request_text = request_data.decode('utf-8')
    print(request_text)

    # Parse request
    lines = request_text.split("\r\n")

    # Select first line
    line = lines[0]

    # Parse fields
    fields = line.split(" ")

    # Parse target
    if(len(fields) < 2):
        continue
    target = fields[1]
    print(target)

    # Respond to target
    if(target == '/'): # The default target, serve 'index.html'
        http_response = bytes(html, 'utf-8')
        client_connection.sendall(http_response)
    else:
        print("huh? I mean 404!")

    # Close client connection
    client_connection.close()
#FIN