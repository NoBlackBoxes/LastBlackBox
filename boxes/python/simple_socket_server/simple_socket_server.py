# Simple HTTP server
import socket

# Load your HTML page (index.html)
path = 'index.html'
file = open(path,"r")
html = file.read()
file.close()

# Set host address (your NB3's IP) and an unused port (1234)
HOST, PORT = '', 1234

# Open a "listening" socket (waits for external connections)
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

    # (Optional) parse request
    #   - if you want to serve different files based on the request

    # Respond to target (send the bytes of your HTML file)
    header = "HTTP/1.1 200 OK\n" 
    http_response = bytes(header+html, 'utf-8')
    client_connection.sendall(http_response)

    # Close client connection
    client_connection.close()

#FIN