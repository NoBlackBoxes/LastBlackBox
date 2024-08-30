# Intelligence

...or not so much.

----

<details><summary><b>Materials</b></summary><p>

Contents|Level|Description| # |Data|Link|
:-------|:---:|:----------|:-:|:--:|:--:|
NPU|10|Coral EdgeTPU USB Type-C|1|[-D-](_data/datasheets/Coral-USB-Accelerator-datasheet.pdf)|[-L-](https://coral.ai/products/accelerator/)
M3 screw (16)|10|16 mm long M3 screw with phillips socket|4|-|-
M3 nut (square)|10|square M3 nut 1.8 mm thick|4|-|[-L-](https://www.accu.co.uk/flat-square-nuts/21326-HFSN-M3-A2)

</p></details>

----

## Topics

- To Do

----

## Goals

### Grey

1. Setup your NPU.
2. Run the demos.

### White

1. Train a new network to detect only your face.
2. Run a keyword to detector to control your robot with voice commands.

----

## NB3

This box will contribute the following (red) components to your NB3

<p align="center">
<img src="_data/images/NB3_intelligence.png" alt="NB3 stage" width="400" height="400">
<p>

----

## Setup your NPU

Your nerual processing unit is made by Google (and distributed by Coral). It contains an EdgeTPU (tensor processing unit) that very efficiently implements the computations used in (feed forward) neural networks. It can connect to your RPi via USB3, allowing you to send "input" data and retreive "outputs" after network inference very quickly. However, in order to communicate with your EdgeTPU, you will need to install some additional libraries.

1. Add the EdgeTPU runtime library "package feed" to the list of feeds that APT will monitor

```bash
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
```

2. Add a security key, which is used to check the integrity of any packages downloaded from this new feed

```bash
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
```

3. Update APT sources

```bash
sudo apt-get update
```

4. Install the "standard" speed EdgeTPU runtime (you could also install the "max" speed version...but be careful, the device will get *HOT!*)

```bash
sudo apt-get install libedgetpu1-std
```

5. Install the Tensorflow-lite interpreter for Python (3.7). You will use this to control your EdgeTPU from Python.

```bash
sudo pip3 install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl
```

6. Now connect your EdgeTPU to the USB3 port (a blue one) of your RPi.

## Test your NPU

Let's see if your EdgeTPU is working.

1. Create a new folder to store some example code and pre-trained neural networks (as well as test datasets)

```bash
cd ~/LastBlackBox
mkdir -p tools
cd tools
mkdir coral
cd coral
git clone https://github.com/google-coral/tflite.git
```

2. Download a "model" that was trained to take pictures of birds and classify them into species

```bash
cd ~/LastBlackBox/tools/coral
cd tflite/python/examples/classification
sudo ./install_requirements.sh
```

3. Run the model (classify_image.py) with a test image, on your EdgeTPU, and see if it works.

```bash
sudo python3 classify_image.py \
--model models/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite \
--labels models/inat_bird_labels.txt \
--input images/parrot.jpg
```


## Cortex

Is great, and confusing.

### Exercise: Face

Find yourself.

----
