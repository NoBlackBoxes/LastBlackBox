# LBB secure server (uses SSL)
import os
import socket
import ssl
import mimetypes

# Get LBB root
LBBROOT = os.environ.get('LBBROOT')

# Get LBB site path
LBBSITE = LBBROOT + "/repo/site/lastblackbox.training"

# Load LBB home page (index.html)
path = LBBSITE + '/index.html'
file = open(path,"r")
html = file.read()
file.close()

# Load LBB style sheet (style.css)
path = LBBSITE + '/style.css'
file = open(path,"r")
style = file.read()
file.close()

# Load LBB banner (banner.png)
path = LBBSITE + '/kit/design/logo/banner.png'
file = open(path,"rb")
banner = file.read()
banner = bytes(banner)
file.close()

# Set host and port
HOST, PORT = '', 8443

# Create SSL context and load private key
context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
pem_path = LBBROOT + "/lbb.pem"
key_path = LBBROOT + "/key.pem"
context.load_cert_chain(pem_path, key_path)

# Open socket
listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listen_socket.bind((HOST, PORT))
listen_socket.listen(5)
print(f'Serving HTTPS on port {PORT} ...')
with context.wrap_socket(listen_socket, server_side=True) as ssock:
    while True:
        client_connection, client_address = ssock.accept()
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

        if(target == '/'):
            http_response = bytes(html, 'utf-8')
            client_connection.sendall(http_response)
        elif(target == '/style.css'):
            http_response = bytes(style, 'utf-8')
            client_connection.sendall(http_response)
        elif(target == '/banner.png'):
            content_type = mimetypes.guess_type(target)[0] or 'text/html'
            print(content_type)

            response = b"HTTP/1.1 200 OK\r\nServer: whatever\r\nDate: Wed, 18 Oct 2017 14:19:11 GMT\r\nContent-Type: image/png\r\nContent-Length: 19115\r\n\r\n"
            # Needs to end with an extra break \r\n
            http_response = response + banner
            client_connection.sendall(http_response)
        elif(target == '/design/box/layout.png'):
            image = """
            HTTP/1.1 200 OK
            Content-Type: image/png

            4412439898693269
            """
            http_response = bytes(image, 'utf-8')
            client_connection.sendall(http_response)
        else:
            print("huh?")
        client_connection.close()
#FIN