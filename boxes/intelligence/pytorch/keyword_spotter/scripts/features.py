# Extract MEL features from audio snippets
import os
import shutil
import random
import wave
import numpy as np
import matplotlib.pyplot as plt
import NB3.Sound.utilities as Utilities

# Specify paths
username = os.getlogin()
LBB = '/home/' + username + '/NoBlackBoxes/LastBlackBox'
project_folder = LBB + '/boxes/intelligence/pytorch/keyword_spotter'
dataset_folder = project_folder + '/_tmp/dataset'
output_folder = project_folder + '/_tmp/features'

# Create (or clear if it exists) output folder
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder)

# Feature plotting function
def plot_feature(name, wav_path, output_path):

    # Load sound from dataset
    wav_obj = wave.open(wav_path)
    num_frames = wav_obj.getnframes()
    byte_data = wav_obj.readframes(num_frames)
    sound = np.frombuffer(byte_data, dtype=np.int16)
    wav_obj.close()
    sound_f = sound.astype(np.float32) / 32768.0

    # Generate Mel Matrix (for audio processing)
    mel_matrix = Utilities.generate_mel_matrix(16000, 40) # 40 Mel Coeffs

    # Compute Mel Spectrogram
    mel_spectrogram = Utilities.compute_mel_spectrogram(sound_f, 640, 320, mel_matrix)

    # Display
    plt.subplot(2,1,1)
    plt.plot(sound_f)
    plt.title(f"Mel Feature Spectrogram: \"{name}\"")
    plt.xlim(0, 16000)
    plt.ylabel('Sound Pressure')
    plt.xticks(range(1000,17000,2000))
    plt.yticks([])
    plt.subplot(2,1,2)
    plt.imshow(mel_spectrogram, aspect='auto', extent=[0, 16000, 0, 40])
    plt.xlabel('Time')
    plt.ylabel('Mel Frequency Bin')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# ------------------------------------------------------
# Plot features for one example of each class in dataset
# ------------------------------------------------------

# Find all Class folders
class_folders = []
for f in os.listdir(dataset_folder):
    if os.path.isdir(dataset_folder + '/' + f):
        if f != '_background_noise_':
            class_folders.append(dataset_folder + '/' + f)

# Create example figure for each class
for class_folder in class_folders:
    class_name = os.path.basename(class_folder)
    wav_names = os.listdir(class_folder)
    wav_paths = []
    for wav_name in wav_names:
        wav_paths.append(class_folder + '/' + wav_name)
    random.shuffle(wav_paths) 
    wav_path = wav_paths[0]     # Select one random example from each class
    output_path = f"{output_folder}/{class_name}_feature.png"
    plot_feature(class_name, wav_path, output_path)
    output_wav_path = f"{output_folder}/{class_name}_feature.wav"
    shutil.copyfile(wav_path, output_wav_path)

#FIN