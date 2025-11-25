# Bootcamp : Signal Processing
Let's learn how to process one-dimensional time series data...like audio waveforms.

<details><summary><i>Materials</i></summary><p>

Name|Description| # |Package|Data|Link|
:-------|:----------|:-----:|:-:|:--:|:--:|
NB3 Ear|I2S mems microphone breakout board|2|Circuit Boards|[-D-](/boxes/audio/NB3_ear)|[-L-](VK)
NB3 Mouth|I2S DAC-AMP breakout board|1|Circuit Boards|[-D-](/boxes/audio/NB3_mouth)|[-L-](VK)
Speaker (Hi-Fi)|3 Watt 4 Ohm with Dupont 2.54 mm socket (High Fidelity: 2831/3128)|1|Large (100)|[-D-](/boxes/audio/_resources/datasheets/3128_3W_4Ohm.jpg)|[-L-](https://www.amazon.co.uk/gp/product/B0D9QXW5FF)
Speaker Mount|Custom laser cut mount for speaker|1|Acrylic Mounts|[-D-](/boxes/audio/-)|[-L-](VK)
Speaker Frame|Custom laser cut frame for speaker|1|Acrylic Mounts|[-D-](/boxes/audio/-)|[-L-](VK)
M3 standoff (15/PS)|15 mm long plug-to-socket M3 standoff|2|Mounting Hardware|[-D-](/boxes/audio/-)|[-L-](https://uk.farnell.com/ettinger/05-13-151/spacer-m3x15-vzk/dp/1466726)
M3 nut (square)|square M3 nut 1.8 mm thick|2|Mounting Hardware|[-D-](/boxes/audio/-)|[-L-](https://www.accu.co.uk/flat-square-nuts/21326-HFSN-M3-A2)
M3 bolt (6)|6 mm long M3 bolt|2|Mounting Hardware|[-D-](/boxes/audio/-)|[-L-](https://www.accu.co.uk/pozi-pan-head-screws/500113-SPP-M3-6-ST-BZP)
M2.5 bolt (6)|6 mm long M2.5 bolt|2|Mounting Hardware|[-D-](/boxes/robotics/)|[-L-](https://www.accu.co.uk/pozi-pan-head-screws/9255-SPP-M2-5-6-A2)
M2.5 nut|regular M2.5 nut|2|Mounting Hardware|[-D-](/boxes/power/-)|[-L-](https://www.accu.co.uk/hexagon-nuts/456430-HPN-M2-5-C8-Z)
M2 bolt (8)|8 mm long M2 bolt|2|Mounting Hardware|[-D-](/boxes/audio/)|[-L-](https://www.accu.co.uk/pozi-pan-head-screws/500101-SPP-M2-8-ST-BZP)
M2 nut|regular M2 nut|2|Mounting Hardware|[-D-](/boxes/audio/)|[-L-](https://www.accu.co.uk/hexagon-nuts/456429-HPN-M2-C8-Z)

</p></details><hr>

## Audio
### NB3 : Streaming Sound
> Let's stream live audio from your NB3's camera to any web browser.


# Project
#### Watch this video: [Signal Processing](https://vimeo.com/manage/videos/1139975157)
<p align="center">
<a href="https://vimeo.com/manage/videos/1139975157" title="Control+Click to watch in new tab"><img src="../../../../boxes/audio/_resources/lessons/thumbnails/Signal-Processing.gif" alt="Signal Processing" width="480"/></a>
</p>

> One-dimensional (1D) signals are common in the world. They often arise as time series (a sequence of measurements made at a regular "sample rate"). Here we will explore techniques to process a common 1D signal...**sound**. We will both measure and generate different sounds. We will develop tools in Python to compare their amplitude (volume), frequency content, and visualize the "spectrogram" of different spoken words (e.g. "Go" vs "Stop").

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

