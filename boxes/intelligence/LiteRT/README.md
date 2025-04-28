# Intelligence : LiteRT
LiteRT is a framework for running inference (trained models) on small devices. It used to be called Tensforflow-Lite.

## Setup
- Acticvate the LBB virtual environment

```bash
pip install tflite-runtime
```
- The current version is 2.14.0

## Test

- Download example data, model, and labels

```bash
# Create a temporary directory
cd ${LBB}/boxes/intelligence/LiteRT/
mkdir _tmp
cd _tmp

# Download and extract photo
wget https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/examples/label_image/testdata/grace_hopper.bmp

# Download and extract model
wget https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz 
tar -xf mobilenet_v1_1.0_224.tgz

# Download and extract labels
wget https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_1.0_224_frozen.tgz
tar -xf mobilenet_v1_1.0_224_frozen.tgz
mv mobilenet_v1_1.0_224/labels.txt .
```
