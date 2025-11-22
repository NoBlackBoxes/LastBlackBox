import time
import subprocess
import io
import numpy as np
import pyaudio
import wave
import soundfile
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
        if sound.ndim == 1: # If mono and 1D, make it (N, 1)
            sound = sound[:, np.newaxis]
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

        # Get WAV info (works with WAVE_EXTENSIBLE etc.)
        info = soundfile.info(wav_path)
        wav_num_channels = info.channels
        wav_sample_rate = info.samplerate
        wav_num_samples = info.frames

        # Try to infer "sample width" in bytes from subtype (for your validation print)
        subtype = info.subtype  # e.g. 'PCM_16', 'PCM_24', 'PCM_32', 'FLOAT', 'DOUBLE'
        if subtype in ("PCM_U8", "PCM_S8"):
            wav_sample_width = 1
        elif subtype == "PCM_16":
            wav_sample_width = 2
        elif subtype == "PCM_24":
            wav_sample_width = 3
        elif subtype in ("PCM_32", "FLOAT"):
            wav_sample_width = 4
        elif subtype == "DOUBLE":
            wav_sample_width = 8
        else:
            wav_sample_width = None  # unknown / exotic

        # Validate WAV file
        if wav_num_channels != self.num_channels:
            print(f"WAV file has inconsistent number of channels {wav_num_channels} for output device {self.num_channels}.")
        if wav_sample_rate != self.sample_rate:
            print(f"WAV file has inconsistent sample rate {wav_sample_rate} for output device {self.sample_rate}.")
        if wav_sample_width != self.sample_width:
            print(f"WAV file has inconsistent sample width {wav_sample_width} for output device {2}.")

        # Read WAV data as float32; always_2d gives shape (frames, channels)
        float_data, sr = soundfile.read(wav_path, dtype="float32", always_2d=True)

        # Limit number of frames to complete buffers
        wav_num_samples = wav_num_samples - (wav_num_samples % self.buffer_size_samples)

        # Truncate to whole buffers
        float_data = float_data[:wav_num_samples, :]

        # Write WAV data to speaker
        self.write(float_data)

        return
    
    # Generate speech using classic Text-to-Speech (TTS) engine (espeak)
    def speak(self, text, voice="en-gb", wpm=140, amp=100, gap=0, pitch=50):
        # Generate speech (send to stdout)
        cmd = ["espeak-ng", "--stdout", f"-v{voice}", f"-s{wpm}", f"-a{amp}", f"-g{gap}", f"-p{pitch}", text]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        wav_bytes = proc.stdout

        # Parse WAV bytes into header + data from memory
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            wav_num_channels = wav_file.getnchannels()
            wav_sample_rate = wav_file.getframerate()
            wav_sample_width = wav_file.getsampwidth()
            wav_num_samples =  wav_file.getnframes()

            # Limit number of frames to complete buffers
            wav_num_samples = wav_num_samples - (wav_num_samples % self.buffer_size_samples)

            # Extract WAV data
            wav_data = wav_file.readframes(wav_num_samples)
        
        # Convert WAV data to float32
        mono_int16 = np.frombuffer(wav_data, dtype=np.int16)
        mono = mono_int16.astype(np.float32) * 3.0517578125e-05
        
        # Resample data to speaker sample rate
        ratio = self.sample_rate / wav_sample_rate
        num_out = int(len(mono) * ratio)
        t_src = np.linspace(0, len(mono)/wav_sample_rate, len(mono), endpoint=False)
        t_dst = np.linspace(0, len(mono)/wav_sample_rate, num_out, endpoint=False)
        mono_data = np.interp(t_dst, t_src, mono).astype(np.float32)

        # Format channels for speaker: mono or stereo, etc.
        if self.num_channels == 2:
            data = np.column_stack((mono_data, mono_data))
        else:
            data = mono_data

        # Write WAV data to speaker
        self.write(data)

        return

    # Check if for sound output is finished
    def is_playing(self):
        if self.current_sample < self.max_samples:
            return True
        else:
            return False

#FIN