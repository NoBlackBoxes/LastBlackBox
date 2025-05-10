# Extract MEL features from audio snippets
import os
import numpy as np
import wave
import matplotlib.pyplot as plt
import NB3.Sound.utilities as Utilities

# Get user name
username = os.getlogin()

# Specify paths
repo_path = '/home/' + username + '/NoBlackBoxes/LastBlackBox'
project_path = repo_path + '/boxes/intelligence/pytorch/keyword_spotter'
dataset_path = project_path + '/_tmp/dataset'
wav_path = dataset_path + '/five/0a2b400e_nohash_0.wav'

# Load example sound from dataset
wav_obj = wave.open(wav_path)
num_frames = wav_obj.getnframes()
byte_data = wav_obj.readframes(num_frames)
sound = np.frombuffer(byte_data, dtype=np.int16)
wav_obj.close()
sound_f = sound.astype(np.float32) / 32768.0

# Generate Mel Matrix (for audio processing)
mel_matrix = Utilities.generate_mel_matrix(16000, 40) # 40 Mel Coeffs

# Compute Mel Spectrogram
mel_spectrogram = Utilities.compute_mel_spectrogram(sound, 640, 320, mel_matrix)

# Display
print(mel_spectrogram.shape)
plt.title("Mel Feature Spectrogram: \"Five\"")
plt.subplot(2,1,1)
plt.plot(sound_f)
plt.xlim(0, 16000)
plt.ylabel('Sound (normalized)')
plt.subplot(2,1,2)
plt.imshow(mel_spectrogram.T, aspect='auto', extent=[0, 16000, 0, 40])
plt.xlabel('Time')
plt.tight_layout()
plt.show()

#FIN