# Benchmark model performance
import os
import wave
import time
import numpy as np
import matplotlib.pyplot as plt
import NB3.Sound.utilities as Utilities

import torch
torch.backends.quantized.engine = 'qnnpack'

# Locals libs
import dataset
#import model_dnn as model
#import model_cnn as model
import model_dscnn as model

# Reimport
import importlib
importlib.reload(dataset)
importlib.reload(model)

# Specify paths
username = os.getlogin()
LBB = '/home/' + username + '/NoBlackBoxes/LastBlackBox'
project_folder = LBB + '/boxes/intelligence/pytorch/keyword_spotter'
dataset_folder = project_folder + '/_tmp/dataset'
features_folder = project_folder + '/_tmp/features'
output_folder = project_folder + '/_tmp/benchmarks'

# Create output folder (if it does not exist)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Run model
def run_model(model, name, wav_path):

    # Load sound from dataset
    wav_obj = wave.open(wav_path)
    num_frames = wav_obj.getnframes()
    byte_data = wav_obj.readframes(num_frames)
    sound = np.frombuffer(byte_data, dtype=np.int16)
    wav_obj.close()
    sound_f = sound.astype(np.float32) / 32768.0

    # Start timer
    start = time.perf_counter()

    # Generate Mel Matrix (for audio processing)
    mel_matrix = Utilities.generate_mel_matrix(16000, 40) # 40 Mel Coeffs

    # Compute Mel Spectrogram
    mel_spectrogram = Utilities.compute_mel_spectrogram(sound_f, 640, 320, mel_matrix)

    # Convert ndarray to Tensor
    features = np.expand_dims(mel_spectrogram, 0)
    features_tensor = torch.from_numpy(features).to(dtype=torch.float32).unsqueeze(0)
    features_tensor = features_tensor.to('cpu')

    # Run model
    output = model(features_tensor)
    end = time.perf_counter()

    inference_time_ms = (end - start) * 1000
    print(f"Inference time: {inference_time_ms:.3f} ms")

    return output

# ---------------
# Benchmark model
# ---------------

# Reload saved quantized model
model_path = model_path = project_folder + '/_tmp/quantized/quantized.pt'
quantized_model = torch.jit.load(model_path, map_location=torch.device('cpu'))
quantized_model.eval()

# Limit CPU resources
torch.set_num_threads(2)

# Find all test files
wav_paths = []
for f in os.listdir(features_folder):
    if f[-3:] == "wav":
        wav_paths.append(f"{features_folder}/{f}")

# Benchmark examples from each class
for wav_path in wav_paths:
    class_name = wav_path.split('_')[0]
    output = run_model(quantized_model, class_name, wav_path)

#FIN