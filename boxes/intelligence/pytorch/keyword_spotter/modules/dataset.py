import os
import torch
import wave
import random
import numpy as np
import NB3.Sound.utilities as Utilities

# Set parameters
sample_rate = 16000
num_window_samples = 640
num_hop_samples = 320
num_mfcc = 40
num_times = 49
silence_magnitude = 0.01
noise_addition_factor = 0.3

# Specify words
non_words = ["silence", "unknown"]
command_words = ["yes", "no", "on", "off", "up", "down", "left", "right", "go", "stop"]
distraction_words = ["backward", "eight", "five", "follow", "forward", "one", "four", "seven", "six", "learn", "nine", "bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila", "tree", "wow", "three", "two", "visual",  "zero"]
classes = non_words + command_words

# Define dataset class (which extends the utils.data.Dataset module)
class custom(torch.utils.data.Dataset):
    def __init__(self, wav_paths, targets, noise, augment=False):
        self.wav_paths = wav_paths
        self.targets = targets
        self.noise = noise
        self.augment = augment
        self.mel_matrix = Utilities.generate_mel_matrix(sample_rate, num_mfcc)

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, idx):
        wav_path = self.wav_paths[idx]
        target = self.targets[idx]

        # Load Sound (from WAV file or Noise/Silence)
        if target == 0: # Silence
            start_frame = random.randint(0, len(self.noise) - sample_rate)
            sound = np.random.normal(0, silence_magnitude, sample_rate).astype(np.float32)
        else:
            sound = load_wav(wav_path)
            if len(sound) < sample_rate:
                buffer = np.zeros(sample_rate - len(sound))
                sound = np.concatenate([sound, buffer])
            
        # Augment?
        if self.augment:
            # 1. Random noise addition
            start_frame = random.randint(0, len(self.noise) - sample_rate)
            noise = self.noise[start_frame:(start_frame+sample_rate)]
            noise_multiplier = random.uniform(0.0, noise_addition_factor)
            sound = sound + (noise_multiplier * noise)

            # 2. Random time shift
            shift = random.randint(-int(0.1 * sample_rate), int(0.1 * sample_rate)) # shift up to +/- 10% of 1 second
            if shift > 0:
                sound = np.pad(sound, (shift, 0), mode='constant')[:sample_rate]
            elif shift < 0:
                sound = np.pad(sound, (0, -shift), mode='constant')[:sample_rate]

            # 3. Random amplitude scaling
            scale = random.uniform(0.8, 1.2)
            sound = sound * scale

        # Compute Features
        mel_spectrogram = Utilities.compute_mel_spectrogram(sound, num_window_samples, num_hop_samples, self.mel_matrix)

        # Convert to int8 range (-127 to 127): appears to help with quantization
        mel_spectrogram *= 127.0

        # Add channel dimension
        features = np.expand_dims(mel_spectrogram, 0)

        # Convert to Float32 (input) and Long (target)
        features = np.float32(features)
        target = np.long(target)

        return features, target

# Load WAV
def load_wav(path):
        wav_obj = wave.open(path)
        num_frames = wav_obj.getnframes()
        byte_data = wav_obj.readframes(num_frames)
        sound = np.frombuffer(byte_data, dtype=np.int16)
        wav_obj.close()
        sound_f = sound.astype(np.float32) / 32768.0
        return sound_f

# Load dataset
def prepare(dataset_folder, split):

    # Find all Class folders
    class_folders = []
    for f in os.listdir(dataset_folder):
        if os.path.isdir(dataset_folder + '/' + f):
            if f != '_background_noise_':
                class_folders.append(dataset_folder + '/' + f)

    # Find all WAV files for each Class
    wav_paths = []
    targets = []
    for class_folder in class_folders:
        paths = os.listdir(class_folder)
        #print(f"{os.path.basename(class_folder)}: {len(paths)} files") 
        full_paths = []
        for path in paths:
            full_paths.append(class_folder + '/' + path)
        random.shuffle(full_paths) 
        num_paths = len(full_paths)
        if os.path.basename(class_folder) in command_words:
            wav_paths.extend(full_paths)
            targets.extend([os.path.basename(class_folder)] * num_paths)
        else:
            # Add a subset of unknown examples (do NOT unbalance the dataset)
            subset_paths = num_paths // 10
            wav_paths.extend(full_paths[:subset_paths])
            targets.extend(["unknown"] * subset_paths)

    # Load all Noise files
    noise_arrays = []
    for f in os.listdir(f"{dataset_folder}/_background_noise_"):
        if f.endswith("wav"):
            noise_path = f"{dataset_folder}/_background_noise_/{f}"
            sound = load_wav(noise_path)
            noise_arrays.append(sound)
    noise_data = np.concatenate(noise_arrays)

    # Include examples for "Silence"
    num_random = int(len(wav_paths) / len(command_words))
    for i in range(num_random):
        wav_paths.append("silence")
        targets.append("silence")

    # Determine target class IDs
    target_list = []
    for t in targets:
        target_index = classes.index(t)
        target_list.append(target_index)

    # Convert to arrays
    wav_paths = np.array(wav_paths)
    target_array = np.array(target_list)

    # Split train/test
    num_samples = len(targets)
    num_train = int(num_samples * split)
    num_test = num_samples - num_train
    indices = np.arange(num_samples)
    shuffled = np.random.permutation(indices)
    train_indices = shuffled[:num_train]
    test_indices = shuffled[num_train:]

    # Bundle
    train_data = (wav_paths[train_indices], target_array[train_indices])
    test_data = (wav_paths[test_indices], target_array[test_indices])

    return train_data, test_data, noise_data

#FIN