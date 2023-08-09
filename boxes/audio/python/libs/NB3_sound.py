import numpy as np
import pyaudio
import wave
from threading import Thread

#
# Utilities
#
def list_devices():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')

    print("\n\nInput Devices\n-----------------\n")
    for i in range(0, numdevices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print(" - Devices id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))
    print("\nOutput Devices\n-----------------\n")
    for i in range(0, numdevices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxOutputChannels')) > 0:
            print(" - Devices id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))
    p.terminate()
    print("-----------------\n\n")
    return

#
# Sound input thread (microphone)
#
class microphone:
    def __init__(self, device, num_channels, sample_rate, format, buffer_size_samples, max_samples):        
        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self.format = format
        self.buffer_size_samples = buffer_size_samples
        self.max_samples = max_samples
        self.valid_samples = 0
        self.freq_bins = np.fft.fftfreq(buffer_size_samples, 1.0/self.sample_rate)[1:]

        # Get pyaudio object
        self.pya = pyaudio.PyAudio()

        # Open audio input stream (from default device)
        self.stream = self.pya.open(input_device_index=device, format=format, channels=num_channels, rate=sample_rate, input=True, output=False, frames_per_buffer=buffer_size_samples)

        # Create rolling buffer (assumes 16-bit signed int samples)
        self.sound = np.zeros((self.max_samples, self.num_channels), dtype=np.int16)
        self.streaming = False

        # Create Speech parameters
        self.speech = False
        self.no_speech_since = 0

        # Create WAV parameters
        self.wav_path = ''
        self.wav_file = 0
        self.wav_max_samples = 0
        self.wav_current_samples = 0
        self.wav_recording = False

        # Configure thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        
    # Start thread method
    def start(self):
        self.streaming = True
        self.thread.start()

    # Update thread method
    def update(self):
        while True :
            # End?
            if self.streaming is False :
                break
            
            # Read raw data and append
            raw_data = self.stream.read(self.buffer_size_samples, exception_on_overflow = False)

            # Write to WAV?
            if self.wav_recording:
                self.wav_file.writeframes(raw_data)
                self.wav_current_samples += (len(raw_data) / 2) # Hack, this is raw bytes, divide by sample byte size (int 16)
                if self.wav_current_samples >= self.wav_max_samples:
                    self.stop_recording()

            # Seperate channel data
            channel_data = np.reshape(np.frombuffer(raw_data, dtype=np.int16), (-1,2))

            # Fill buffer...and then concat
            if self.valid_samples < self.max_samples:
                self.sound[self.valid_samples:(self.valid_samples + self.buffer_size_samples), :] = channel_data
                self.valid_samples = self.valid_samples + self.buffer_size_samples
            else:
                self.sound = np.vstack([self.sound[self.buffer_size_samples:, :], channel_data])
                self.valid_samples = self.max_samples

            # Is it speech or not? - Compute power in speech range for current buffer...if big change...speech...if not...silence...
            amplitudes = np.abs(np.fft.fft(channel_data[:,0]))[1:]
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
            if(speech_ratio > 0.25):
                self.no_speech_since = 0
            else:
                self.no_speech_since += 1

            # Is there speaking recently?
            if self.no_speech_since > 10:
                self.speech = False
            else:
                self.speech = True

        # Shutdown thread
        self.stream.stop_stream()
        self.stream.close()
        self.pya.terminate()

    # Read sound method
    def read(self):
        num_valid_samples = self.valid_samples
        self.valid_samples = 0
        return self.sound[:num_valid_samples]

    # Read most recent samples method
    def read_latest(self, num_samples):
        if(self.valid_samples < num_samples):
            return self.sound[:num_samples, :]
        else:
            return self.sound[(self.valid_samples-num_samples):self.valid_samples, :]

    # Reset sound input
    def reset(self):
        self.sound = np.zeros((self.max_samples, self.num_channels), dtype=np.int16)
        self.valid_samples = 0
        return

    # Start saving WAV
    def start_recording(self, wav_path, wav_max_duration):
        self.wav_path = wav_path
        self.wav_max_duration = wav_max_duration

        # Prepare a WAV file
        self.wav_file = wave.open(wav_path, 'wb')
        self.wav_file.setnchannels(self.num_channels)
        self.wav_file.setsampwidth(2)
        self.wav_file.setframerate(self.sample_rate)

        # Start recording
        self.wav_current_duration = 0
        self.wav_recording = True

        return

    # Stop saving WAV
    def stop_recording(self):
        
        # Close WAV file
        self.wav_file.close()

        # Stop recording
        self.wav_current_duration = 0
        self.wav_recording = False

        return

    # Recording?
    def is_recording(self):
        return self.wav_recording

    # Speaking?
    def is_speaking(self):
        return self.speech

    # Stop thread method
    def stop(self):
        self.streaming = False

#
# Sound output thread (speaker)
#
class speaker:
    def __init__(self, device, num_channels, sample_rate, format, buffer_size_samples):        
        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self.format = format
        self.buffer_size_samples = buffer_size_samples
        self.current_sample = 0
        self.max_samples = 0

        # Get pyaudio object
        self.pya = pyaudio.PyAudio()

        # Open audio output stream (from default device)
        self.stream = self.pya.open(output_device_index=device, format=format, channels=num_channels, rate=sample_rate, input=False, output=True, frames_per_buffer=buffer_size_samples)

        # Set params
        self.streaming = False

        # Configure thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        
    # Start thread method
    def start(self):
        self.streaming = True
        self.thread.start()

    # Update thread method
    def update(self):
        empty_buffer = np.zeros((self.buffer_size_samples, self.num_channels), dtype=np.int16)
        while True :
            # End?
            if self.streaming is False :
                break
            
            # Playing?
            if self.current_sample < self.max_samples:
                # Write sound data buffer
                channel_data = self.sound[self.current_sample:(self.current_sample + self.buffer_size_samples), :]
                raw_data = channel_data[:]
                self.stream.write(raw_data, self.buffer_size_samples * self.num_channels, exception_on_underflow = False)

                # Increment buffer position
                self.current_sample = self.current_sample + self.buffer_size_samples
            else:
                self.stream.write(empty_buffer, self.buffer_size_samples, exception_on_underflow = False)
        
        # Shutdown thread
        self.stream.stop_stream()
        self.stream.close()
        self.pya.terminate()

    # Write sound method
    def write(self, sound):
        num_samples = np.shape(sound)[0]
        max_samples = num_samples + (self.buffer_size_samples - (num_samples % self.buffer_size_samples))
        self.sound = np.zeros((max_samples, self.num_channels), dtype=np.int16)
        self.sound = np.copy(sound)
        self.current_sample = 0
        self.max_samples = max_samples
        return

    # Reset sound output
    def reset(self):
        self.current_sample = 0
        return

    # Check if for sound output is finished
    def playing(self):
        if self.current_sample < self.max_samples:
            return True
        else:
            return False

    # Stop thread method
    def stop(self):
        self.streaming = False