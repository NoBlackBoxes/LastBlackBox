# Intelligence : NPU : Coral
A neural processing unit based on Goggle EdgeTPU.

## Setup your NPU
Your neural processing unit is made by Google (and distributed by Coral). It contains an EdgeTPU (tensor processing unit) that very efficiently implements the computations used in (feed forward) neural networks. It connects to your RPi via USB3, allowing you to send "input" data and retrieve "outputs" of the neural network inference very quickly. However, in order to communicate with your EdgeTPU, you will need to install some additional libraries.

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

5. Install the Tensorflow-lite interpreter for Python. You will use this to control your EdgeTPU from Python.

```bash
python3 -m pip install tflite-runtime
```

6. Update UDEV rules to allow USB doral device access.

- Create file "/etc/udev/rules.d/71-edgetpu.rules"

```bash
sudo nano /etc/udev/rules.d/71-edgetpu.rules
```

- Add the following contents and save the file

```txt
SUBSYSTEMS=="usb", ATTRS{idVendor}=="1a6e", ATTRS{idProduct}=="089a", MODE="0664", TAG+="uaccess"
```

- Reload UDEV rules
```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```

7. Now connect your EdgeTPU to the USB3 port (a blue one) of your RPi.

## Test your NPU

Let's see if your EdgeTPU is working!

1. Create a new folder to store some example code and pre-trained neural networks (as well as test datasets)

```bash
# Navigate to your LBB temporary folder
cd ~/NoBlackBoxes/LastBlackBox/_tmp

# Make and enter a directory to store the example code and datasets
mkdir coral
cd coral

# Clone and enter the Google tflite examples
git clone https://github.com/google-coral/tflite.git
cd tflite
```

2. Download a "model" that was trained to take pictures of birds and classify them into species

```bash
cd python/examples/classification
./install_requirements.sh
```

3. Run the model (classify_image.py) with a test image, on your EdgeTPU, and see if it works.

```bash
python3 classify_image.py \
--model models/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite \
--labels models/inat_bird_labels.txt \
--input images/parrot.jpg
```

