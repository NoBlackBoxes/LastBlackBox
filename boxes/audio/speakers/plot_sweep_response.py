import numpy as np
import matplotlib.pyplot as plt
import wave

# Specify sweep paths
sweep_paths = [
    "/home/kampff/Dropbox/Adam/Music/debugging/3watt_sweep.wav",
    "/home/kampff/Dropbox/Adam/Music/debugging/5watt_sweep.wav",
    "/home/kampff/Dropbox/Adam/Music/debugging/small_sweep.wav"
]
sweep_labels = ["3 Watt", "5 Watt", "Small (3 W)"]

# Load sweeps
plt.figure()
for sweep_path in sweep_paths:
    # Read WAV file
    sweep_wave = wave.open(sweep_path, 'rb')
    num_channels = sweep_wave.getnchannels()
    sample_rate = sweep_wave.getframerate()
    sample_width = sweep_wave.getsampwidth()
    num_samples =  sweep_wave.getnframes()
    sweep_binary = sweep_wave.readframes(num_samples)
    sweep_wave.close()

    # Process sound
    if sample_width == 4: 
        data_type = np.int32
    else:
        data_type = np.int16
    sweep_data = np.frombuffer(sweep_binary, dtype=data_type)
    trigger_sample = np.where(sweep_data > 2e8)[0][0]
    sweep_data = sweep_data[trigger_sample:]
    plt.plot(sweep_data, alpha=0.5)
plt.legend(sweep_labels)    
plt.show()
