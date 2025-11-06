# The Last Black Box : Audio
In this box, you will learn about audio...

## Audio
Listen up, speak up.

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

### Microphones
> MEMs mics and beyond.


### Speakers
> Better sounding speakers.


### I2S
> I2S is a fully-digital protocol used by modern CPUs to talk to audio devices over a small number of connections. It requires only a clock, LR select (for selecting which channel is being written/read), data in (for microphone input), and data out (for speaker output).


### Signal Processing
> One-dimensional (1D) signals are common in the world. They often arise as time series (a sequence of measurements made a regular "sample rate"). Here we will explore techniques to process one of these 1D signals...**sound**. We will measure sound amplitude (volume), timing of sound arrival at two different microphones (ears), and the frequency content of different sound types.

- Measure volume
- - Measure inter-ear timing
- Measure frequency (FFT)
- Measure spectrogram
- Distinguish Go vs Stop (?)

# Project
### NB3 : Build and Install the Linux I2S Driver
> Let's build (compile) and install the Linux device driver for our NB3's sound card. This software module is required to access the NB3's ears (microphone) and mouth (speaker).

<details><summary><weak>Guide</weak></summary>
:-:-: A video guide to completing this project can be viewed <a href="https://vimeo.com/1042781850" target="_blank" rel="noopener noreferrer">here</a>.
</details><hr>


### NB3 : Install the Ears
> Let's add some ears to your NB3.

<details><summary><weak>Guide</weak></summary>
:-:-: A video guide to completing this project can be viewed <a href="https://vimeo.com/1042943195" target="_blank" rel="noopener noreferrer">here</a>.
</details><hr>


### NB3 : Install the Mouth
> Let's add a mouth to your NB3.

<details><summary><weak>Guide</weak></summary>
:-:-: A video guide to completing this project can be viewed <a href="https://vimeo.com/1042947561" target="_blank" rel="noopener noreferrer">here</a>.
</details><hr>


