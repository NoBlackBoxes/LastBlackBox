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
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind((ip_address, port))
sock.listen(1)
print(f"\n🌐 HTTP Server running at http://localhost:{port}")
print(f"    - \"Control + C\" to Quit -")

# Serve incoming connections
try:
    while True:                             # This loop will keep checking for a connection
        conn, addr = sock.accept()          # Accept a connection request (waits until one is received)
        print(f"Connected to by {addr}")

        try:
            # When a connection arrives, retrieve/decode HTTP request and send response
            request = conn.recv(1024)
            request_text = request.decode('utf-8')
            print(f"{len(request)} bytes received:\n{request_text}")

            # (Optional) parse the HTTP request
            #   - if you want to serve different files based on the content of the request

            # Respond to target (send the bytes of your HTML file after a "success" header)
            header = "HTTP/1.1 200 OK\n" 
            response = bytes(header+html, 'utf-8')
            conn.sendall(response)

            # Close client connection
            conn.close()

        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            print("Client disconnected.")
        finally:
            print("Connection closed; returning to accept new clients.")

except KeyboardInterrupt:
    print("\nShutting down...")

finally:
    sock.close()

#FIN