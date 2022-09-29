import numpy as np
import serial
import pyaudio
import time

# Configure serial port
ser = serial.Serial()
ser.baudrate = 19200
ser.port = '/dev/ttyUSB0'
ser.open()
time.sleep(2.00) # Wait for connection before sending any data

# Congure audio recording
CHUNK = 4800                # Buffer size
FORMAT = pyaudio.paInt32    # Data type
CHANNELS = 2                # Number of channels
RATE = 48000                # Sample rate (Hz)
RECORD_SECONDS = 500        # Duration

# Get pyaudio object
p = pyaudio.PyAudio()

# Open audio stream (from default device)
stream = p.open(format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK) 

# Dance
shuffle = False
l_speed = 0
r_speed = 0
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):

    # Read raw data and append
    raw_data = stream.read(CHUNK)

    # Convert to numpy array
    interleaved_data = np.frombuffer(raw_data, dtype=np.int32)
    
    # Extract left and right values
    left = interleaved_data[::2] 
    right = interleaved_data[1::2]  

    # DO SOME PROCESSING HERE #

    # Report volume (on left)
    scale = 65536
    volume = np.mean(np.abs(left))/scale 
    print("L: {0:.6f}, R: {1:.6f}".format(np.mean(np.abs(left))/scale, np.mean(np.abs(right))/scale))
    if(volume > 255):
        volume =255.0

    # Sende message
    if (volume > 100):
        l_direction = 0
        r_direction = 0
        l_speed = volume
        r_speed = volume
    else:
        l_direction = 0
        r_direction = 0
        l_speed = l_speed*.95
        r_speed = r_speed*.95

    # Send message
    message = bytearray([int(l_direction), int(l_speed), int(r_direction), int(r_speed)])
    ser.write(message)
    time.sleep(0.010)

# Shutdown
ser.close()
stream.stop_stream()
stream.close()
p.terminate()
