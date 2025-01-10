# Build a Brain : Artificial Intelligence
What is intelligence? Can we create it in a machine? Can you create it in your **NB3**?

## Audio
Listen up, speak up.

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

### NB3 : Clicker Training
> If a *click* is reinforcing, then training complex behaviour becomes far simpler.


## Intelligence
...or not so much.

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

