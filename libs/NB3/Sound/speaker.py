import time
import numpy as np
import pyaudio
import wave
from threading import Lock

#
# Sound output (speaker)
#
class Speaker:
    def __init__(self, device, num_channels, format, sample_rate, buffer_size_samples):        
        self.num_channels = num_channels
        self.format = format
        self.sample_rate = sample_rate
        self.buffer_size_samples = buffer_size_samples
        self.current_sample = 0
        self.max_samples = 0
        self.volume = 0.25
        self.mutex = Lock()

        # Set format
        if format == 'int16':
            self.format = pyaudio.paInt16
            self.dtype = np.int16
            self.sample_width = 2
        elif (format == 'int32'):
            self.format = pyaudio.paInt32
            self.dtype = np.int32
            self.sample_width = 4
        else:
            print("(NB3.Sound) Unsupported output sample format")
            exit(-1)

        # Create buffers
        self.empty = np.zeros((self.buffer_size_samples, self.num_channels), dtype=np.float32)
        self.output_data = np.zeros((self.buffer_size_samples, self.num_channels), dtype=np.float32)
        self.integer_data = np.zeros((self.buffer_size_samples, self.num_channels), dtype=self.dtype)
        self.sound = np.zeros((self.max_samples, self.num_channels), dtype=np.float32)

        # Profiling
        self.callback_count = 0
        self.callback_accum = 0.0
        self.callback_max = 0.0

        # Configure callback
        def callback(input_data, frame_count, time_info, status):

            # Profiling
            start_time = time.clock_gettime(time.CLOCK_REALTIME)

            # Lock thread
            with self.mutex:

                # How many samples remain to output?
                remaining_samples = self.max_samples - self.current_sample
                
                # Output a full buffer, partial buffer, or empty buffer
                if remaining_samples >= self.buffer_size_samples:
                    output_start_sample = self.current_sample
                    output_stop_sample = self.current_sample + self.buffer_size_samples
                    self.output_data = np.reshape(self.sound[output_start_sample:output_stop_sample, :], -1)
                    self.current_sample += self.buffer_size_samples
                elif remaining_samples > 0:
                    output_start_sample = self.current_sample
                    output_stop_sample = self.max_samples
                    final_buffer  = np.copy(self.empty)
                    final_buffer[0:remaining_samples, :] = self.sound[output_start_sample:output_stop_sample, :]
                    self.output_data = np.reshape(final_buffer, -1)
                    self.current_sample += remaining_samples
                else:
                    self.output_data = np.reshape(self.empty, -1)

            # Convert from float to sample format
            if self.sample_width == 2:
                self.integer_data = np.int16(self.output_data * 2**15)
            else:
                self.integer_data = np.int32(self.output_data * 2**31)

            # Profiling
            stop_time = time.clock_gettime(time.CLOCK_REALTIME)
            self.callback_count += 1
            duration = stop_time-start_time
            if duration > self.callback_max:
                self.callback_max = duration
            self.callback_accum += duration

            return (self.integer_data, pyaudio.paContinue)

        # Get pyaudio object
        self.pya = pyaudio.PyAudio()
        print("\n\n\n")

        # Open audio output stream (from default device)
        self.stream = self.pya.open(output_device_index=device, format=self.format, channels=num_channels, rate=sample_rate, input=False, output=True, frames_per_buffer=buffer_size_samples, start=False, stream_callback=callback)


    # Start streaming
    def start(self):
        self.stream.start_stream()

    # Stop streaming
    def stop(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pya.terminate()

    # Write sound method
    def write(self, sound):
        num_samples = np.shape(sound)[0]
        max_samples = num_samples - (num_samples % self.buffer_size_samples)
        self.sound = np.zeros((max_samples, self.num_channels), dtype=np.float32)
        self.sound = np.copy(sound[:max_samples,:]) * self.volume
        self.current_sample = 0
        self.max_samples = max_samples
        return

    # Clear sound output
    def clear(self):
        self.current_sample = 0
        self.max_samples = 0
        return

    # Reset sound output
    def reset(self):
        self.current_sample = 0
        return

    # Play WAV file
    def play_wav(self, wav_path):
        self.wav_path = wav_path

        # Read a WAV file
        wav_file = wave.open(wav_path, 'rb')
        wav_num_channels = wav_file.getnchannels()
        wav_sample_rate = wav_file.getframerate()
        wav_sample_width = wav_file.getsampwidth()
        wav_num_samples =  wav_file.getnframes()

        # Validate WAV file
        if wav_num_channels != self.num_channels:
            print(f"WAV file has inconsistent number of channels {wav_num_channels} for output device {self.num_channels}.")
        if wav_sample_rate != self.sample_rate:
            print(f"WAV file has inconsistent sample rate {wav_sample_rate} for output device {self.sample_rate}.")
        if wav_sample_width != 2:
            print(f"WAV file has inconsistent sample width {wav_sample_width} for output device {2}.")

        # Set number of frames
        wav_num_samples = wav_num_samples - (wav_num_samples % self.buffer_size_samples)

        # Read WAV file
        wav_data = wav_file.readframes(wav_num_samples)
        wav_file.close()

        # Separate channel data and convert to float
        if wav_sample_width == 2:
            channel_data = np.reshape(np.frombuffer(wav_data, dtype=np.int16).transpose(), (-1,self.num_channels))
            float_data = np.float32(channel_data) * 3.0517578125e-05
        elif wav_sample_width == 4:
            channel_data = np.reshape(np.frombuffer(wav_data, dtype=np.int32).transpose(), (-1,self.num_channels))
            float_data = np.float32(channel_data) * 4.656612873077393e-10
        else:
            print("(NB3.Sound) Unsupported WAV output sample format")
            exit(-1)
        self.sound = np.copy(float_data) * self.volume

        # Start playing
        self.current_sample = 0
        self.max_samples = wav_num_samples

        return
    
    # Check if for sound output is finished
    def is_playing(self):
        if self.current_sample < self.max_samples:
            return True
        else:
            return False

# FIN