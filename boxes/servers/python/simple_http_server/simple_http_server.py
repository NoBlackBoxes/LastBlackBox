# Simple HTTP Server
# - This server parses the HTTP request to find the "requested resource", and if availabe, it sends that to the client
# - ...otherwise it just sends "index.html"
import socket

# Helper function to load an HTML page into memory
def load_html(path):
    file = open(path,"r")
    html = file.read()
    file.close()
    return html

# Specify IP address and port to use for the socket
ip_address = '' # Leaving this empty means listen on all available interfaces (IP address)
port = 1234

# Open a "listening" socket (waits for external connections)
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind((ip_address, port))
sock.listen(1)
print(f"\n🌐 HTTP Server running at http://localhost:{port}")
print(f"    - \"Control + C\" to Quit -\n")

# Serve incoming connections
try:
    while True:                             # This loop will keep checking for a connection
        conn, addr = sock.accept()          # Accept a connection request (waits until one is received)
        print(f"Connected to by {addr}")

        try:
            # When a connection arrives, retrieve/decode HTTP request
            request = conn.recv(1024)
            request_text = request.decode('utf-8')

            # Parse the HTTP request serve different resources (files) based on the requested content
            first_request_line = request_text.split('\n')[0]            # The first line contains the request type amd resource name
            request_type = first_request_line.split(' ')[0]             # Extract request type, GET, PUT, etc.
            requested_resource = first_request_line.split(' ')[1][1:]   # Extract resource name (removes the leading '/')
            print(f"Request Type: {request_type}\nRequested Resource: {requested_resource}\n")

            # Respond to request (send the bytes of requested HTML file after a "success" header)
            header = "HTTP/1.1 200 OK\n"
            if (requested_resource.endswith("html")):
                html = load_html(requested_resource)
            else:
                html = load_html("index.html")
            response = bytes(header+html, 'utf-8')
            conn.sendall(response)

            # Close client connection
            conn.close()

        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            print("Client disconnected.\n")
        finally:
            print("Connection closed; returning to accept new clients.\n")

except KeyboardInterrupt:
    print("\nShutting down...")

finally:
    sock.close()

#FIN