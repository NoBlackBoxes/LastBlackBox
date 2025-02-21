import time
import numpy as np
import pyaudio
import wave
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
        self.mutex = Lock()
        self.freq_bins = np.fft.fftfreq(self.buffer_size_samples, 1.0/self.sample_rate)[1:]

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

            # Seperate channel data
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
                    self.valid_samples = self.valid_samples + self.buffer_size_samples
                else:
                    self.sound = np.vstack([self.sound[self.buffer_size_samples:, :], self.float_data])
                    self.valid_samples = self.max_samples

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
        return

    # Copy latest sound data
    def latest(self, num_samples):
        if num_samples < self.valid_samples:
            latest = np.copy(self.sound[(self.valid_samples-num_samples):self.valid_samples, :])
        else:
            latest = np.copy(self.sound[0:self.valid_samples, :])
        return latest

    # Speaking?
    def is_speech(self):
        # Is there data for speech detection?
        if self.valid_samples < self.buffer_size_samples:
            return False

        # Get latest
        latest = np.copy(self.sound[(self.valid_samples-self.buffer_size_samples):self.valid_samples, :])

        # Compute amps and energies
        amplitudes = np.abs(np.fft.fft(latest[:,0]))[1:]
        energies = amplitudes**2

        # Compute total energy
        energy_per_freq = {}
        for (i, freq) in enumerate(self.freq_bins):
            if abs(freq) not in energy_per_freq:
                energy_per_freq[abs(freq)] = energies[i] * 2
        total_energy = sum(energy_per_freq.values())

        # Compute voice energy
        voice_energy = 0
        for f in energy_per_freq.keys():
            if 300 < f < 3000:                      # Human voice range
                voice_energy += energy_per_freq[f]

        # Compute speech ratio
        speech_ratio = voice_energy/total_energy

        # Is there speaking now?
        if(speech_ratio > 0.5):
            speech = True
        else:
            speech = False
        
        return speech

    # Mel features
    def mel_features(self, window_size):
        # Is there data for mel feature detection?
        if self.valid_samples < self.buffer_size_samples:
            return False

        latest = self.sound[(self.valid_samples-self.buffer_size_samples):self.valid_samples, 0]

        window_length_samples = int(0.025 * self.sample_rate)
        hop_length_samples = int(0.010 * self.sample_rate)
        fft_length = 512

        frames = []
        for i in range(0, len(latest) - window_length_samples + 1, hop_length_samples):
            frame = latest[i:i+window_length_samples]
            windowed = frame * np.hanning(window_length_samples)
            frames.append(np.abs(np.fft.rfft(windowed, fft_length)))
        spectrogram = np.stack(frames)

        mel_matrix = np.zeros((spectrogram.shape[1], 20))
        freq_bins = np.linspace(0, self.sample_rate / 2, spectrogram.shape[1])
        mel_bins = np.linspace(125, 3800, 22)

        for i in range(20):
            lower = mel_bins[i]
            center = mel_bins[i + 1]
            upper = mel_bins[i + 2]
            mel_matrix[:, i] = np.maximum(0, np.minimum((freq_bins - lower) / (center - lower), (upper - freq_bins) / (upper - center)))

        mel_spectrogram = np.dot(spectrogram, mel_matrix)
        log_mel_spectrogram = np.log(mel_spectrogram + 1e-6)

        return log_mel_spectrogram.mean(axis=0)

    # Start saving WAV
    def save_wav(self, wav_path, wav_max_samples):

        # Prepare WAV file
        wav_file = wave.open(wav_path, 'wb')
        wav_file.setnchannels(self.num_channels)
        wav_file.setsampwidth(2)    # int16 WAV
        wav_file.setframerate(self.sample_rate)

        # Determine WAV range (in samples)
        if wav_max_samples < self.valid_samples:
            wav_start_sample = self.valid_samples - wav_max_samples
            wav_stop_sample = self.valid_samples
        else:
            wav_start_sample = 0
            wav_stop_sample = self.valid_samples

        # Convert to integer (16-bit)
        integer_data = np.int16(self.sound[wav_start_sample:wav_stop_sample,:] * 2**15)

        # Convert sound to frame data
        frames = np.reshape(integer_data, -1)

        # Write to WAV
        wav_file.writeframes(frames)

        # Close WAV file
        wav_file.close()

        return

# FIN