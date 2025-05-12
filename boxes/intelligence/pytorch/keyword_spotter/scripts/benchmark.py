# Benchmark model performance
import os
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
output_folder = project_folder + '/_tmp/benchmarks'

# Create output folder (if it does not exist)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Run model
def run_model(name, wav_path):

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

    






    # Output
    output = 1.0

    return output

# ---------------
# Benchmark model
# ---------------

# Find all Class folders
class_folders = []
for f in os.listdir(dataset_folder):
    if os.path.isdir(dataset_folder + '/' + f):
        if f != '_background_noise_':
            class_folders.append(dataset_folder + '/' + f)

# Benchmark examples from each class
for class_folder in class_folders:
    class_name = os.path.basename(class_folder)
    wav_names = os.listdir(class_folder)
    wav_paths = []
    for wav_name in wav_names:
        wav_paths.append(class_folder + '/' + wav_name)
    random.shuffle(wav_paths) 
    wav_path = wav_paths[0]     # Select one random example from each class
    output = run_model(class_name, wav_path)
    print(output)

#FIN