import pyaudio
import numpy as np
from scipy.fft import rfft, rfftfreq
import serial


CHUNK_SIZE = 4096           # Buffer size
FORMAT = pyaudio.paInt16    # Data type
CHANNELS = 2                # Number of channels
RATE = 22050                # Sample rate (Hz)


def detect_whistling():
    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    # Open the audio stream
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK_SIZE)

    # Open serial connection to send instructions to the arduino
    ser = serial.Serial('/dev/ttyUSB0', 9600)
    try:
        ser.open()
    except:
        ser.close()
        ser.open()

    try:
        while True:
            # Read audio data from the stream
            raw_data = stream.read(CHUNK_SIZE)

            # Detect whistling
            interleaved_data = np.frombuffer(raw_data, dtype=np.int16)
            print(interleaved_data)
            left = interleaved_data[::2] 
            #right = interleaved_data[1::2]  

            # calculate Fourier transform
            fourier = np.abs(rfft(left))
            freqs = rfftfreq(len(left), 1 / RATE)

            # find the frequency with the highest amplitude
            max_amplitude = np.max(fourier)
            max_amplitude_index = np.where(fourier == max_amplitude)[0][0]
            peak_frequency = freqs[max_amplitude_index]
            print(peak_frequency)

            whistling_frequency = 1070 # this is the frequency you determined with the recorded audio file

            # compare the current peak-frequency with the whistling_frequency
            offset = 20  # define an offset to use a window of frequencies
            if whistling_frequency - offset < peak_frequency and whistling_frequency + offset > peak_frequency:
                # Send intructions to the robot
                message = b'o'
                ser.write(message)
            else:
                message = b'x'
                ser.write(message)

    except KeyboardInterrupt:
        # Close the stream and socket when interrupted
        stream.stop_stream()
        stream.close()
        ser.close()

if __name__ == '__main__':
    detect_whistling()