import librosa
import numpy as np
from scipy.fft import rfft, rfftfreq

import matplotlib.pyplot as plt


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
    max_amplitude_index = np.argmax(fourier)
    frequency = freqs[max_amplitude_index]

    # print the frequency
    print(f"The dominant frequency is {int(frequency)} Hz")

    plt.figure()
    plt.plot(freqs, fourier, 'k-', lw=1)
    plt.scatter(freqs[max_amplitude_index], fourier[max_amplitude_index], s=100)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Fourier Transform')
    plt.show()


if __name__ == '__main__':
    # Argparse the file path to the audio file containing the whistling
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='Path to the audio file')
    args = parser.parse_args()

    # call the function
    determine_whistling_frequency(args.path)