import librosa
import numpy as np
from scipy.fft import rfft, rfftfreq


def determine_whistling_frequency(path):
    """
    This function will determine the frequency of the whistling
    """

    # load the audio file
    audio, rate = librosa.load(path, sr=None, mono=True)

    # calculate Fourier transform
    fourier = np.abs(rfft(audio))
    freqs = rfftfreq(len(audio), 1 / rate)

    # find the frequency with the highest amplitude
    max_amplitude = np.max(fourier)
    max_amplitude_index = np.where(fourier == max_amplitude)[0][0]
    frequency = freqs[max_amplitude_index]

    # print the frequency
    print(f"The frequency of the whistling is {int(frequency)} Hz")


if __name__ == '__main__':
    # Argparse the file path to the audio file containing the whistling
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='Path to the audio file')
    args = parser.parse_args()

    # call the function
    determine_whistling_frequency(args.path)