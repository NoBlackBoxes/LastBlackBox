# TTS voice synthesis request
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
HOST, PORT = '192.168.1.109', 4321

# Define helper functions
def recv_msg(sock):
    # Read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]

    # Read the message data
    return recvall(sock, msglen)

def recvall(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

# Connect and send
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    s.sendall(b"I'm sorry, Dave, I am afraid I can't do that.")

    # Receive WAV data
    data = recv_msg(s)

    # Write wav file
    file = open(output_wav_path, "wb")
    wav_data = file.write(data)
    file.close()    

print(f"Received {len(data)}")

#FIN