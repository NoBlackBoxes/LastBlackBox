# Build a Brain : Artificial Intelligence
What is intelligence? Can we create it in a machine? Can you create it in your **NB3**?

## Audio
Listen up, speak up.

<details><summary><i>Materials</i></summary><p>

Contents|Depth|Description| # |Data|Link|
:-------|:---:|:----------|:-:|:--:|:--:|
NB3 Ear|01|I2S mems microphone breakout board|2|[-D-](/boxes/audio/NB3_ear)|[-L-](VK)
NB3 Mouth|01|I2S DAC-AMP breakout board|1|[-D-](/boxes/audio/NB3_mouth)|[-L-](VK)
Speaker|01|3 Watt 4 Ohm with Dupont 2.54 mm socket|1|[-D-](/boxes/audio/_resources/datasheets/CQR0508.jpg)|[-L-](https://www.amazon.co.uk/dp/B09JJ28D6R)
Speaker (Hi-Fi)|01|3 Watt 4 Ohm with Dupont 2.54 mm socket (High Fidelity: 2831/3128)|1|[-D-](/boxes/audio/_resources/datasheets/3128_3W_4Ohm.jpg)|[-L-](https://www.amazon.co.uk/gp/product/B0D9QXW5FF)
Speaker Mount|01|Custom laser cut mount for speaker|1|[-D-](/boxes/audio/-)|[-L-](VK)
Speaker Frame|01|Custom laser cut frame for speaker|1|[-D-](/boxes/audio/-)|[-L-](VK)
M3 standoff (15/PS)|01|15 mm long plug-to-socket M3 standoff|2|[-D-](/boxes/audio/-)|[-L-](https://uk.farnell.com/ettinger/05-13-151/spacer-m3x15-vzk/dp/1466726)
M3 nut (square)|01|square M3 nut 1.8 mm thick|2|[-D-](/boxes/audio/-)|[-L-](https://www.accu.co.uk/flat-square-nuts/21326-HFSN-M3-A2)
M3 bolt (6)|01|6 mm long M3 bolt|2|[-D-](/boxes/audio/-)|[-L-](https://www.accu.co.uk/pozi-pan-head-screws/500113-SPP-M3-6-ST-BZP)
M2.5 bolt (6)|01|6 mm long M2.5 bolt|2|[-D-](/boxes/audio/)|[-L-](https://www.accu.co.uk/pozi-pan-head-screws/9255-SPP-M2-5-6-A2)
M2.5 nut|01|regular M2.5 nut|2|[-D-](/boxes/audio/-)|[-L-](https://www.accu.co.uk/hexagon-nuts/456430-HPN-M2-5-C8-Z)
M2 bolt (8)|01|8 mm long M2 bolt|2|[-D-](/boxes/audio/)|[-L-](https://www.accu.co.uk/pozi-pan-head-screws/500101-SPP-M2-8-ST-BZP)
M2 nut|01|regular M2 nut|2|[-D-](/boxes/audio/)|[-L-](https://www.accu.co.uk/hexagon-nuts/456429-HPN-M2-C8-Z)

</p></details><hr>

### Microphones
> MEMs mics and beyond.


### I2S
> I2S is a fully-digital protocol used by modern CPUs to talk to audio devices over a small number of connections. It requires only a clock, LR select (for selecting which channel is being written/read), data in (for microphone input), and data out (for speaker output).


#### Watch this video: [NB3 : Build and Install the Linux I2S Driver](https://vimeo.com/1042781850)
> Let's build (compile) and install the Linux device driver for our NB3's sound card. This software module is required to access the NB3's ears (microphone) and mouth (speaker).


#### Watch this video: [NB3 : Install the Ears](https://vimeo.com/1042943195)
> Let's add some ears to your NB3.


### Speakers
> Better sounding speakers.


#### Watch this video: [NB3 : Install the Mouth](https://vimeo.com/1042947561)
> Let's add a mouth to your NB3.


### Signal Processing
> Digital signal processing is a set of techniques for processing one-dimensional data (1D).


## Vision
Look at everyone.

<details><summary><i>Materials</i></summary><p>

Contents|Depth|Description| # |Data|Link|
:-------|:---:|:----------|:-:|:--:|:--:|
Camera (RPi v3)|01|RPi color camera with auto-focus (version 3)|1|[-D-](/boxes/vision/_resources/datasheets/rpi_camera_v3.pdf)|[-L-](https://uk.farnell.com/raspberry-pi/sc0872/rpi-camera-mod-3-standard-lens/dp/4132318)
NB3 Camera Mount|01|Custom laser cut mount for RPi camera|1|[-D-](/boxes/vision/NB3_camera_mount)|[-L-](VK)
NB3 Cortex Mount|01|Custom laser cut holder for NPU|1|[-D-](/boxes/vision/NB3_cortex_mount)|[-L-](VK)
M2.5 bolt (6)|01|6 mm long M2.5 bolt|4|[-D-](/boxes/vision/)|[-L-](https://www.accu.co.uk/pozi-pan-head-screws/9255-SPP-M2-5-6-A2)
M2.5 standoff (20/PS)|01|20 mm long plug-to-socket M2.5 standoff|4|[-D-](/boxes/vision/)|[-L-](https://uk.farnell.com/wurth-elektronik/971200151/standoff-hex-male-female-20mm/dp/2884418)
M3 nut (square)|01|square M3 nut 1.8 mm thick|1|[-D-](/boxes/vision/)|[-L-](https://www.accu.co.uk/flat-square-nuts/21326-HFSN-M3-A2)
M3 bolt (12)|01|12 mm long M3 bolt|1|[-D-](/boxes/vision/)|[-L-](https://www.accu.co.uk/pozi-pan-head-screws/500116-SPP-M3-12-ST-BZP)
M2 bolt (8)|01|8 mm long M2 bolt|4|[-D-](/boxes/vision/)|[-L-](https://www.accu.co.uk/pozi-pan-head-screws/500101-SPP-M2-8-ST-BZP)
M2 nut|01|regular M2 nut|4|[-D-](/boxes/vision/)|[-L-](https://www.accu.co.uk/hexagon-nuts/456429-HPN-M2-C8-Z)

</p></details><hr>

### Cameras
> From CCDs to CMOS.


#### Watch this video: [NB3 : Install the Eye](https://vimeo.com/1042945461)
> Let's install a camera on your NB3.

- Test your camera with the following command
```bash
# Navigate to LBB root
cd ~/NoBlackBoxes/LastBlackBox

# Create folder for images in the temporary folder
mkdir _tmp/images

# Navigate to the temporary image folder
cd _tmp/images

# Capture a still image and save it to a file
libcamera-still -o test.png
```


### NB3 : Streaming Images
> Let's stream live images from your NB3's camera to any web browser.


### Image-Processing
> Old-school (classical) techniques for processing two-dimensional data (2D).


## Learning
Supervised, unsupervised, and reinforcement learning

<details><summary><i>Materials</i></summary><p>

Contents|Depth|Description| # |Data|Link|
:-------|:---:|:----------|:-:|:--:|:--:|

</p></details><hr>

### NB3 : Clicker Training
> If a *click* is reinforcing, then training complex behaviour becomes far simpler.


## Intelligence
...or not so much.

<details><summary><i>Materials</i></summary><p>

Contents|Depth|Description| # |Data|Link|
:-------|:---:|:----------|:-:|:--:|:--:|

</p></details><hr>

### Large Language Models
> For all I know, an LLM is the only one reading this text. Hi ChatGPT, godspeed.


# Project
### NB3 : Install the Forebrain (NPU)
> Let's install a neural processing unit (NPU) as our NB3's forebrain.

<details><summary><weak>Guide</weak></summary>
:-:-: A video guide to completing this project can be viewed <a href="NPU)" target="_blank" rel="noopener noreferrer">here</a>.
</details><hr>

- Your neural processing unit is made by Google (and distributed by Coral). It contains an EdgeTPU (tensor processing unit) that very efficiently implements the computations used in (feed forward) neural networks. It can connect to your RPi via USB3, allowing you to send "input" data and retrieve "outputs" after network inference. However, in order to communicate with your EdgeTPU, you will need to install some additional libraries.
- Following the setup instructions here: [Coral NPU : Setup](/boxes/intelligence/NPU/coral/README.md)

