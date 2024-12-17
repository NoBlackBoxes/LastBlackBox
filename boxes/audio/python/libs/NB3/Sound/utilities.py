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
                device_id = p.get_device_info_by_host_api_device_index(0, i).get('id')
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
                device_id = p.get_device_info_by_host_api_device_index(0, i).get('id')
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

# FIN