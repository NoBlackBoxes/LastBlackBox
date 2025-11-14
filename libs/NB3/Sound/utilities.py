import sys
import pyaudio
import numpy as np
import scipy.signal as signal

#
# Utilities
#

# List input and output devices
def list_devices():
    """
    List PyAudio devices (input and output)
    """
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    num_devices = info.get('deviceCount')

    print("\n\nInput Devices\n-----------------\n")
    for i in range(0, num_devices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print(" - Devices id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))
    print("\nOutput Devices\n-----------------\n")
    for i in range(0, num_devices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxOutputChannels')) > 0:
            print(" - Devices id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))
    print("-----------------\n\n")
    p.terminate()
    return

# Get input device by name
def get_input_device_by_name(name):
    """
    Get PyAudio input device starting with specified name
    """
    p = pyaudio.PyAudio()
    print("\n\n\n")
    info = p.get_host_api_info_by_index(0)
    num_devices = info.get('deviceCount')
    device_id = -1
    for i in range(0, num_devices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            device_name = p.get_device_info_by_host_api_device_index(0, i).get('name')
            if device_name.startswith(name):
                device_id = i
    p.terminate()
    if device_id == -1:
        print(f"Device starting with \"{name}\" not found. Select a different audio input device.")
    return device_id

# Get output device by name
def get_output_device_by_name(name):
    """
    Get PyAudio output device starting with specified name
    """
    p = pyaudio.PyAudio()
    print("\n\n\n")
    info = p.get_host_api_info_by_index(0)
    num_devices = info.get('deviceCount')
    device_id = -1
    for i in range(0, num_devices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxOutputChannels')) > 0:
            device_name = p.get_device_info_by_host_api_device_index(0, i).get('name')
            if device_name.startswith(name):
                device_id = i
    p.terminate()
    if device_id == -1:
        print(f"Device starting with \"{name}\" not found. Select a different audio output device.")
    return device_id

# Generate pure tone
def generate_pure_tone(duration, frequency, sample_rate, num_channels):
    """
    Generate a pure tone (sine wave) of specific frequency and duration
    """
    data = np.sin(2.0 * np.pi * np.arange(sample_rate*duration) * (frequency / sample_rate))
    if num_channels == 2:
        sound = np.vstack((data, data)).T
    else:
        sound = data
    return sound.astype(np.float32)

# Generate square wave
def generate_square_wave(duration, frequency, sample_rate, num_channels, duty_cycle=0.5):
    """
    Generate a square wave of specific frequency, duration, and duty cycle
    """
    t = np.arange(int(sample_rate * duration)) / sample_rate
    
    # Sawtooth phase: 0 → 1 repeating
    phase = (t * frequency) % 1.0

    # Square wave: high if phase < duty cycle
    data = np.where(phase < duty_cycle, 1.0, -1.0).astype(np.float32)

    # Stereo / mono formatting
    if num_channels == 2:
        sound = np.vstack((data, data)).T
    else:
        sound = data
    return sound.astype(np.float32)

def generate_note(duration, fundamental, sample_rate, num_channels=1, num_harmonics=10):
    """
    Generate a harmonic-rich note from a fundamental frequency.
    """
    num_samples = int(sample_rate * duration)

    # Nyquist limit
    nyquist = sample_rate / 2.0

    # Compute amplitudes: 1/n^2 rolloff (more "instrument-esque"" than 1/n)
    amplitudes = 1.0 / (np.arange(1, num_harmonics + 1) ** 2)

    # Build the note as a sum of harmonics
    note = np.zeros(num_samples, dtype=np.float32)
    for n in range(1, num_harmonics + 1):
        frequency = n * fundamental
        if frequency > nyquist:
            break  # stop if harmonic frequency exceeds Nyquist
        tone = generate_pure_tone(duration, frequency, sample_rate, num_channels=1)
        note += amplitudes[n - 1] * tone
    max_val = np.max(np.abs(note))
    if max_val > 0:
        note = note / max_val

    # Format channels
    if num_channels == 2:
        sound = np.vstack((note, note)).T
    else:
        sound = note
    return sound.astype(np.float32)

# Generate frequency sweep
def generate_frequency_sweep(duration, start_frequency, stop_frequency, sample_rate, num_channels):
    """
    Generate a linear frequency sweep of specific duration duration
    """
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    data = signal.chirp(t, f0=start_frequency, f1=stop_frequency, t1=duration, method='linear').astype(np.float32)
    if num_channels == 2:
        sound = np.vstack((data, data)).T
    else:
        sound = data
    return sound

# Generate Mel Cepstral Coefficient Matrix
def generate_mel_matrix(sample_rate, num_mfcc):
    fft_length = 512
    mel_matrix = np.zeros((fft_length // 2 + 1, num_mfcc))
    freq_bins = np.linspace(0, sample_rate / 2, fft_length // 2 + 1)
    freq_bins_mel = 1127.0 * np.log(1.0 + freq_bins / 700.0)
    mel_bins = np.linspace(1127.0 * np.log(1.0 + 60 / 700.0), 1127.0 * np.log(1.0 + 3800 / 700.0), num_mfcc + 2)
    for i in range(num_mfcc):
        lower = mel_bins[i]
        center = mel_bins[i + 1]
        upper = mel_bins[i + 2]
        mel_matrix[:, i] = np.maximum(0, np.minimum((freq_bins_mel - lower) / (center - lower), (upper - freq_bins_mel) / (upper - center)))
    return mel_matrix

# Compute Mel Spectrogram
def compute_mel_spectrogram(sound, window_samples, hop_samples, mel_matrix):
    fft_length = 512

    # Compute spectrograms
    frames = []
    for i in range(0, len(sound) - window_samples + 1, hop_samples):
        frame = sound[i:i+window_samples]
        windowed = frame * np.hanning(window_samples)
        frames.append(np.abs(np.fft.rfft(windowed, fft_length)))
    spectrogram = np.stack(frames)

    # Apply mel filters and take log
    mel_spectrogram = np.dot(spectrogram, mel_matrix)
    log_mel_spectrogram = np.log(mel_spectrogram + 0.0001)

    # Normalise
    log_mel_spectrogram -= np.mean(log_mel_spectrogram, axis=0, keepdims=True)
    log_mel_spectrogram /= (3 * np.std(log_mel_spectrogram, axis=0, keepdims=True))
    log_mel_spectrogram = np.clip(log_mel_spectrogram, -1.0, 1.0).astype(np.float32)

    return log_mel_spectrogram.T # freq bins x times

def compute_spectrum(sound, sample_rate, min_freq=10, max_freq=10000, normalize=True):
    # Ensure single channel (Mono)
    if sound.ndim > 1:
        sound = np.mean(sound, axis=1)                  # If multi-channel, average channels

    # Determine frequency output range of FFT
    n = len(sound)                                      # Number of Samples recorded
    freqs = np.fft.rfftfreq(n, d=1/sample_rate)         # Compute the frequency bins based on sample rate (48 kHz) and length of buffer (5 sec)
    if (sample_rate / 2) < max_freq:                    # Limit max frequency to Nyquist Criterion
        max_freq = (sample_rate / 2)
    frange = (freqs >= min_freq) & (freqs <= max_freq)  # Only consider frequencies between 10 Hz and 10 kHz

    # Compute FFT
    fft = np.fft.rfft(sound)                            # FFT (real only)
    magnitudes = np.abs(fft)                            # Compute magnitudes

    # Apply frequency range
    freqs = freqs[frange]
    magnitudes = magnitudes[frange]
 
    # (optional) Normalize FFT
    if normalize and np.max(magnitudes) > 0:
        magnitudes = magnitudes / np.max(magnitudes)

    return freqs, magnitudes

def find_peaks(x, y, min_prominence=0.1, min_separation=10):
    dx = np.abs(x[1] - x[0])
    min_separation_bins = int(min_separation / dx)
    peak_idx, props = signal.find_peaks(y, prominence=min_prominence, distance=min_separation_bins)

    return x[peak_idx], y[peak_idx]

def meter_start():
    print("\n\n")                   # reserve two lines
    sys.stdout.write("\x1b[?25l")   # hide cursor
    sys.stdout.flush()

def meter_update(left, right, width=50):
    left = max(0.0, min(1.0, float(left)))      # Clamp range 0 to 1.0
    right = max(0.0, min(1.0, float(right)))    # Clamp range 0 to 1.0

    # Draw meter bar
    def bar(label, level):
        filled = int(level * width)
        bar = "█" * filled + " " * (width - filled)
        return f"{label:>5} |{bar}| {level*100:5.1f}%"

    # Move cursor up 2 lines, redraw
    sys.stdout.write("\x1b[2A")  # move up two lines
    sys.stdout.write("\x1b[2K" + bar("Left", left) + "\n")
    sys.stdout.write("\x1b[2K" + bar("Right", right) + "\n")
    sys.stdout.flush()

def meter_stop():
    sys.stdout.write("\x1b[?25h\n")  # show cursor again
    sys.stdout.flush()

# -------------- 
# Useful Values
# -------------- 
NOTE_FREQS = {
    # Octave 0
    "C0": 16.35,  "C#0": 17.32, "Db0": 17.32, "D0": 18.35,  "D#0": 19.45, "Eb0": 19.45,
    "E0": 20.60,  "F0": 21.83,  "F#0": 23.12, "Gb0": 23.12, "G0": 24.50,  "G#0": 25.96, "Ab0": 25.96,
    "A0": 27.50,  "A#0": 29.14, "Bb0": 29.14, "B0": 30.87,

    # Octave 1
    "C1": 32.70,  "C#1": 34.65, "Db1": 34.65, "D1": 36.71,  "D#1": 38.89, "Eb1": 38.89,
    "E1": 41.20,  "F1": 43.65,  "F#1": 46.25, "Gb1": 46.25, "G1": 49.00,  "G#1": 51.91, "Ab1": 51.91,
    "A1": 55.00,  "A#1": 58.27, "Bb1": 58.27, "B1": 61.74,

    # Octave 2
    "C2": 65.41,  "C#2": 69.30, "Db2": 69.30, "D2": 73.42,  "D#2": 77.78, "Eb2": 77.78,
    "E2": 82.41,  "F2": 87.31,  "F#2": 92.50, "Gb2": 92.50, "G2": 98.00,  "G#2": 103.83, "Ab2": 103.83,
    "A2": 110.00, "A#2": 116.54, "Bb2": 116.54, "B2": 123.47,

    # Octave 3
    "C3": 130.81, "C#3": 138.59, "Db3": 138.59, "D3": 146.83, "D#3": 155.56, "Eb3": 155.56,
    "E3": 164.81, "F3": 174.61, "F#3": 185.00, "Gb3": 185.00, "G3": 196.00, "G#3": 207.65, "Ab3": 207.65,
    "A3": 220.00, "A#3": 233.08, "Bb3": 233.08, "B3": 246.94,

    # Octave 4 (Middle C = C4, A4=440)
    "C4": 261.63, "C#4": 277.18, "Db4": 277.18, "D4": 293.66, "D#4": 311.13, "Eb4": 311.13,
    "E4": 329.63, "F4": 349.23, "F#4": 369.99, "Gb4": 369.99, "G4": 392.00, "G#4": 415.30, "Ab4": 415.30,
    "A4": 440.00, "A#4": 466.16, "Bb4": 466.16, "B4": 493.88,

    # Octave 5
    "C5": 523.25, "C#5": 554.37, "Db5": 554.37, "D5": 587.33, "D#5": 622.25, "Eb5": 622.25,
    "E5": 659.25, "F5": 698.46, "F#5": 739.99, "Gb5": 739.99, "G5": 783.99, "G#5": 830.61, "Ab5": 830.61,
    "A5": 880.00, "A#5": 932.33, "Bb5": 932.33, "B5": 987.77,

    # Octave 6
    "C6": 1046.50, "C#6": 1108.73, "Db6": 1108.73, "D6": 1174.66, "D#6": 1244.51, "Eb6": 1244.51,
    "E6": 1318.51, "F6": 1396.91, "F#6": 1479.98, "Gb6": 1479.98, "G6": 1567.98, "G#6": 1661.22, "Ab6": 1661.22,
    "A6": 1760.00, "A#6": 1864.66, "Bb6": 1864.66, "B6": 1975.53,

    # Octave 7
    "C7": 2093.00, "C#7": 2217.46, "Db7": 2217.46, "D7": 2349.32, "D#7": 2489.02, "Eb7": 2489.02,
    "E7": 2637.02, "F7": 2793.83, "F#7": 2959.96, "Gb7": 2959.96, "G7": 3135.96, "G#7": 3322.44, "Ab7": 3322.44,
    "A7": 3520.00, "A#7": 3729.31, "Bb7": 3729.31, "B7": 3951.07,

    # Octave 8
    "C8": 4186.01
}

#FIN