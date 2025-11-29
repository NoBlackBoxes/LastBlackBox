# TTS voice synthesis server
import os
import struct
import socket

# Get user name
username = os.getlogin()

# Specify paths
repo_path = '/home/' + username + '/NoBlackBoxes/LastBlackBox'
box_path = repo_path + '/boxes/intelligence/llms/chatNB3/server'
output_wav_path = box_path + '/_tmp/output.wav'

# Set host (this computer) and port (1234)
HOST, PORT = '', 4321

# Define helper functions
def send_msg(sock, msg):
    # Prefix each message with a 4-byte length (network byte order)
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)

# Open socket
listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listen_socket.bind((HOST, PORT))
listen_socket.listen(1)
print(f'Serving Voice on port {PORT} ...')

# Serve incoming connections
while True:
    # Listen for a connection
    client_connection, client_address = listen_socket.accept()

    # When a connection arrives, parse text message
    request_data = client_connection.recv(1024)
    request_text = request_data.decode('utf-8')
    print(request_text)

    # Synthesisze voice response
    cmd = "echo \"{0}\" | piper --model _tmp/en_GB-alan-low.onnx --output_file {1}".format(request_text, output_wav_path)
    result = os.system(cmd)

    # Read WAV output
    file = open(output_wav_path,"rb")
    wav_data = file.read()
    file.close()

    # Send WAV output
    send_msg(client_connection, wav_data)
    
    # Close connection
    client_connection.close()
#FIN