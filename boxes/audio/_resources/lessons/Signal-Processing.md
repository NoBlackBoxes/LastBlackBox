# Audio : Signal Processing
One-dimensional (1D) signals are common in the world. They often arise as time series (a sequence of measurements made at a regular "sample rate"). Here we will explore techniques to process a common 1D signal...**sound**. We will both measure and generate different sounds. We will develop tools in Python to compare their amplitude (volume), frequency content, and visualize the "spectrogram" of different spoken words (e.g. "Go" vs "Stop").

## [Video](https://vimeo.com/manage/videos/1139975157)

## Concepts

## Lesson
- Explore, test, and expand upon these Python examples for processing signals measured from the NB3's microphones (ears): [Measurement](/boxes/audio/signal-processing/python/measurement/)
    - Stream the live frequency spectrum from your NB3 to your PC.
        - Run the [server](/boxes/audio/signal-processing/python/measurement/03_stream_frequency.py) on your NB3 (via SSH)
        - Run the [client](/boxes/audio/signal-processing/python/measurement/clients/live_plot_frequency_client_pyglet.py) on your PC (in a separate VS Code window)
- Explore, test, and expand upon these Python examples for processing signals generated and output to the NB3's speaker (mouth): [Generation](/boxes/audio/signal-processing/python/generation/)
    - *Note*: For text-to-speech synthesis you will need to install the Linux espeak package:
- *code*
```bash
sudo apt install espeak-ng
```
