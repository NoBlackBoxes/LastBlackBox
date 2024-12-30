# Intelligence : NB3 : Install the Forebrain (NPU)
Let's install a neural processing unit (NPU) as our NB3's forebrain.

## [Video]()

## Concepts

## Connections

## Lesson
Your neural processing unit is made by Google (and distributed by Coral). It contains an EdgeTPU (tensor processing unit) that very efficiently implements the computations used in (feed forward) neural networks. It can connect to your RPi via USB3, allowing you to send "input" data and retrieve "outputs" after network inference very quickly. However, in order to communicate with your EdgeTPU, you will need to install some additional libraries.

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

5. Install the PyCoral library (using apt, not pip)

```bash
sudo apt-get install python3-pycoral
```

6. Run a test inference

```bash
# Navigate to your LBB temporary folder
cd ~/NoBlackBoxes/LastBlackBox/_tmp

# Make and enter a directory to store the example code and datasets
mkdir coral
cd coral

# Clone and enter the pycoral GitHub repo
git clone https://github.com/google-coral/pycoral.git
cd pycoral

# Download a test model and dataset
bash examples/install_requirements.sh classify_image.py

# Run the test "classify" model
python3 examples/classify_image.py \
--model test_data/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite \
--labels test_data/inat_bird_labels.txt \
--input test_data/parrot.jpg
```
