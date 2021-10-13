import pyaudio
import wave
import numpy as np

CHUNK = 4096                # Buffer size
FORMAT = pyaudio.paInt16    # Data type
CHANNELS = 2                # Number of channels
RATE = 16000                # Sample rate (Hz)
RECORD_SECONDS = 5          # Duration
WAVE_OUTPUT_FILENAME = "test.wav"

# Get pyaudio object
p = pyaudio.PyAudio()

# Open audio stream (from default device)
stream = p.open(format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK) 

# Append frames of data
frames = []
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    # Read raw data and append
    raw_data = stream.read(CHUNK)
    frames.append(raw_data)
    
    # Convert to numpy array
    interleaved_data = np.frombuffer(raw_data, dtype=np.int16)

    # Extract left and right values
    left = interleaved_data[::2] 
    right = interleaved_data[1::2]  

    # DO SOME PROCESSING HERE #

    # Report volume (on left)
    print("L: {0:.2f}, R: {1:.2f}".format(np.mean(np.abs(left)), np.mean(np.abs(right))))

# Shutdown
stream.stop_stream()
stream.close()
p.terminate()

# Save a wav file
wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()