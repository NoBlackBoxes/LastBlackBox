import librosa
import matplotlib.pyplot as plt
import numpy as np


def plot_spectrogram(path):
    # Load audio file
    audio, rate = librosa.load(path, sr=None)

    # Create a spectrogram
    spectrogram = librosa.stft(audio)

    # Convert to decibel
    spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)

    # Plot the spectrogram
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(spectrogram_db, sr=rate, x_axis='time', y_axis='hz')
    plt.colorbar()
    plt.title('Spectrogram')
    
    # Save the plot
    save_path = path.replace("wav", "png")
    plt.savefig(f"{save_path}")


# run main function
if __name__ == '__main__':
    # use argparse to get the path to the audio file
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='Path to the audio file')
    args = parser.parse_args()
    plot_spectrogram(args.path)

