# Simple HTTP server
import socket

# Load home page data (index_chat.html)
path = "index_chat.html"
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

    # Respond to target
    if(target == '/'): # The default target, serve 'index.html'
        header = "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n"
        response = bytes(header+html, 'utf-8')
        client_connection.sendall(response)

    elif(target == '/favicon.ico'):
        header = "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n"
        response = bytes(header+html, 'utf-8')
        client_connection.sendall(response)

    elif(target.startswith('/message')):
        header = "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n"
        response = bytes(header+html, 'utf-8')
        client_connection.sendall(response)
        message = target.split('=')[1].replace("+", " ")
        print(f"- {message}, [From: {client_address}]")

    else:
        header = "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n"
        body = "404 Not Found"
        response = header.encode() + body.encode()
        client_connection.sendall(response)

    # Close client connection
    client_connection.close()


#FIN