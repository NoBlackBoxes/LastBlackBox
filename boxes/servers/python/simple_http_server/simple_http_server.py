# Simple HTTP Server
import socket
import netifaces

# Load your HTML page (index.html) in memory
path = 'index.html'
file = open(path,"r")
html = file.read()
file.close()

# Specify IP address and port to use for the socket
ip_address = '' # Leaving this empty means listen on all available interfaces (IP address)
port = 1234

# Open a "listening" socket (waits for external connections)
listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listen_socket.bind((ip_address, port))
listen_socket.listen(1)
print(f"\n🌐 HTTP Server running at http://localhost:{port}")
print(f"    - \"Control + C\" to Quit -")

# Serve incoming connections
while True:
    # Listen for a connection
    client_connection, client_address = listen_socket.accept()

    # When a connection arrives, retrieve/decode HTTP request
    request = client_connection.recv(1024)
    request_text = request.decode('utf-8')
    print(f"{len(request)} bytes received:\n{request_text}")

    # (Optional) parse request
    #   - if you want to serve different files based on the content of the request

    # Respond to target (send the bytes of your HTML file after a "success" header)
    header = "HTTP/1.1 200 OK\n" 
    response = bytes(header+html, 'utf-8')
    client_connection.sendall(response)

    # Close client connection
    client_connection.close()

#FIN