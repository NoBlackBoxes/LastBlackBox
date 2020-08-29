# LBB server
import os
import socket
import mimetypes

# Get LBB root
LBBROOT = os.environ.get('LBBROOT')

# Load LBB home page (index.html)
path = LBBROOT + '/repo/server/index.html'
file = open(path,"r")
html = file.read()
file.close()

# Load LBB style sheet (style.css)
path = LBBROOT + '/repo/server/style.css'
file = open(path,"r")
style = file.read()
file.close()

# Load LBB banner (banner.png)
path = LBBROOT + '/repo/server/banner.png'
file = open(path,"rb")
banner = file.read()
banner = bytes(banner)
file.close()

# Set hot and port
HOST, PORT = '', 8888

# Open socket
listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listen_socket.bind((HOST, PORT))
listen_socket.listen(1)
print(f'Serving HTTP on port {PORT} ...')
while True:
    client_connection, client_address = listen_socket.accept()
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