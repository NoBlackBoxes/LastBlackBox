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
    Generate a pure tone (sinewave) of specific frequency and duration
    """
    data = np.sin(2.0 * np.pi * np.arange(sample_rate*duration) * (frequency / sample_rate))
    if num_channels == 2:
        sound = np.vstack((data, data)).T
    else:
        sound = data
    return sound

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


# FIN