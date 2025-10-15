# Socket Client (plot)
#  This program should run on your PC. It will connect to your NB3 and then receive data streamed
# over the socket and plot it in a real-time graph.
import socket
import numpy as np
import matplotlib.pyplot as plt

# Specify host IP address and port to use for the socket.
ip_address = '192.168.1.80'  # Use the IP of your NB3 (the server)
port = 1234  # Use the same port as specified in the socket_server

# Create a Socket that establish the server connection
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(60)  # Timeout after 60 seconds of inactivity
sock.connect((ip_address, port))
print(f"Connected to server at {ip_address}:{port}")

# Setup plot
num_samples_to_plot = 1024
plot_buffer = np.zeros(num_samples_to_plot, dtype=np.uint8)
write_pos = 0

plt.ion()
fig, ax = plt.subplots()
x = np.arange(num_samples_to_plot)
(line,) = ax.plot(x, plot_buffer, lw=1)
ax.set_ylim(0, 255)
ax.set_xlim(0, num_samples_to_plot - 1)
ax.set_xlabel("sample index")
ax.set_ylabel("value (uint8)")
ax.set_title("Analog Input 0")

# The Socket Client Loop
try:
    while True: # Receive 16 bytes of data
        bytes = sock.recv(16)
        if not bytes:
            print("No data received, waiting...")  # Log when no data is received
            continue  # Keep the connection alive
        data = np.frombuffer(bytes, dtype=np.uint8)
        n = len(data)

        # Write data into circular buffer
        end = write_pos + n
        if end <= num_samples_to_plot:
            plot_buffer[write_pos:end] = data
        else:
            k = num_samples_to_plot - write_pos
            plot_buffer[write_pos:] = data[:k]
            plot_buffer[:end - num_samples_to_plot] = data[k:]
        write_pos = (write_pos + n) % num_samples_to_plot

        # Plot data
        line.set_ydata(plot_buffer)
        plt.pause(0.001) # tiny GUI tick; prevents render backlog
        # DEBUG: print(f"Received: {data}")

        # Allow graceful exit by closing the plot window
        if not plt.fignum_exists(fig.number):
            print("Plot closed by user.")
            break

except socket.timeout:
    print("Socket timed out. No data received for 60 seconds.")
except Exception as e:
    print(f"Error: {e}")
finally:
    sock.close()

# FIN