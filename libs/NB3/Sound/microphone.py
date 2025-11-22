import time
import numpy as np
import pyaudio
import soundfile
from threading import Lock

#
# Sound input (microphone)
#
class Microphone:
    def __init__(self, device, num_channels, format, sample_rate, buffer_size_samples, max_samples):        
        self.num_channels = num_channels
        self.format = format
        self.sample_rate = sample_rate
        self.gain = 1.0
        self.buffer_size_samples = buffer_size_samples
        self.max_samples = max_samples
        self.valid_samples = 0
        self.new_samples = 0
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
            print("(NB3.Sound) Unsupported input sample format")
            exit(-1)

        # Create buffers
        self.output_data = np.zeros((0), dtype=np.float32)
        self.channel_data = np.zeros((self.buffer_size_samples, self.num_channels), dtype=self.dtype)
        self.float_data = np.zeros((self.buffer_size_samples, self.num_channels), dtype=np.float32)
        self.sound = np.zeros((self.max_samples, self.num_channels), dtype=np.float32)
        
        # Profiling
        self.callback_count = 0
        self.callback_accum = 0.0
        self.callback_max = 0.0

        # Define callback
        def callback(input_data, frame_count, time_info, status):

            # Profiling
            start_time = time.clock_gettime(time.CLOCK_REALTIME)

            # Separate channel data
            self.channel_data = np.reshape(np.frombuffer(input_data, dtype=self.dtype).transpose(), (-1,self.num_channels))

            # Convert to float
            if self.sample_width == 2:
                self.float_data = np.float32(self.channel_data) * 3.0517578125e-05 * self.gain
            else:
                self.float_data = np.float32(self.channel_data) * 4.656612873077393e-10 * self.gain

            # Lock thread
            with self.mutex:
                # Fill buffer...and then concat
                if self.valid_samples < self.max_samples:
                    self.sound[self.valid_samples:(self.valid_samples + self.buffer_size_samples), :] = self.float_data
                    self.valid_samples += self.buffer_size_samples
                else:
                    self.sound = np.vstack([self.sound[self.buffer_size_samples:, :], self.float_data])
                    self.valid_samples = self.max_samples
                self.new_samples += self.buffer_size_samples

            # Profiling
            stop_time = time.clock_gettime(time.CLOCK_REALTIME)
            self.callback_count += 1
            duration = stop_time-start_time
            if duration > self.callback_max:
                self.callback_max = duration
            self.callback_accum += duration
            
            return (self.output_data, pyaudio.paContinue)

        # Get pyaudio object
        self.pya = pyaudio.PyAudio()
        print("\n\n\n")

        # Open audio input stream
        self.stream = self.pya.open(input_device_index=device, format=self.format, channels=num_channels, rate=sample_rate, input=True, output=False, frames_per_buffer=buffer_size_samples, start=False, stream_callback=callback)

    # Start streaming
    def start(self):
        self.stream.start_stream()

    # Stop streaming
    def stop(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pya.terminate()

    # Reset sound input
    def reset(self):
        self.sound = np.zeros((self.max_samples, self.num_channels), dtype=np.float32)
        self.valid_samples = 0
        self.new_samples = 0
        return

    # Copy latest sound data
    def latest(self, num_samples):
        if num_samples == -1: # Return only new samples
            latest = np.copy(self.sound[(self.valid_samples - self.new_samples):self.valid_samples, :])
        elif num_samples < self.valid_samples:
            latest = np.copy(self.sound[(self.valid_samples-num_samples):self.valid_samples, :])
        else:
            latest = np.copy(self.sound[0:self.valid_samples, :])
        self.new_samples = 0
        return latest

    # Start saving WAV
    def save_wav(self, wav_path, wav_max_samples, subtype="PCM_32"):

        # Determine WAV range (in samples)
        if wav_max_samples < self.valid_samples:
            wav_start_sample = self.valid_samples - wav_max_samples
            wav_stop_sample = self.valid_samples
        else:
            wav_start_sample = 0
            wav_stop_sample = self.valid_samples

        # Slice the recorded sound: shape (samples, channels)
        data = self.sound[wav_start_sample:wav_stop_sample, :]

        # Ensure [-1, 1] range, float32 for soundfile
        data = np.clip(data, -1.0, 1.0).astype(np.float32)

        # For mono: flatten from (N,1) -> (N,)
        if self.num_channels == 1 and data.ndim == 2:
            data = data[:, 0]

        # Write 32-bit signed PCM WAV
        soundfile.write(wav_path, data, self.sample_rate, subtype=subtype)

        return

#FIN