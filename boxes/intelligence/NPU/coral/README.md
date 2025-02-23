# Intelligence : NPU : Coral
A neural processing unit based on Goggle EdgeTPU.

## Setup your NPU
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

## Python Versions
The tflite-runtime on PyPI currently requires Python 3.11. This can make it difficult to run the Coral examples on an up-to-date host system. It is possible to install an earlier version of Python.

## Building (on NB3)
- Install required libraries
```bash
sudo apt install libabsl-dev libusb-1.0-0-dev xxd cmake
```
- Install earlier version of flatbuffers
```bash
git clone https://github.com/google/flatbuffers.git
cd flatbuffers/
git checkout v23.5.26
mkdir build && cd build
cmake .. \
    -DFLATBUFFERS_BUILD_SHAREDLIB=ON \
    -DFLATBUFFERS_BUILD_TESTS=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local
make -j2
sudo make install
cd ../..
```
- Clone tensorflow and checkout desired version: 
```bash
git clone https://github.com/tensorflow/tensorflow
cd tensorflow
git checkout v2.14.0 # (same version of tflite runtime)
cd ..
```
- Clone libedgetpu repo
```bash
git clone https://github.com/feranick/libedgetpu
```
- Update LDFLAGS in Makefile
```make
LIBEDGETPU_LDFLAGS := \
	-Wl,-Map=$(BUILDDIR)/output.map \
	-shared \
	-Wl,--soname,libedgetpu.so.1 \
	-Wl,--version-script=$(BUILDROOT)/tflite/public/libedgetpu.lds \
	-fuse-ld=gold \
	-lflatbuffers \
	-labsl_flags_internal \
	-labsl_flags_marshalling \
	-labsl_str_format_internal \
	-labsl_flags_reflection \
	-labsl_flags_private_handle_accessor \
	-labsl_flags_commandlineflag \
	-labsl_flags_commandlineflag_internal \
	-labsl_flags_config \
	-labsl_flags_program_name \
	-labsl_cord \
	-labsl_cordz_info \
	-labsl_cord_internal \
	-labsl_cordz_functions \
	-labsl_cordz_handle \
	-labsl_hash \
	-labsl_city \
	-labsl_bad_variant_access \
	-labsl_low_level_hash \
	-labsl_raw_hash_set \
	-labsl_bad_optional_access \
	-labsl_hashtablez_sampler \
	-labsl_exponential_biased \
	-labsl_synchronization \
	-labsl_graphcycles_internal \
	-labsl_stacktrace \
	-labsl_symbolize \
	-labsl_debugging_internal \
	-labsl_demangle_internal \
	-labsl_malloc_internal \
	-labsl_time \
	-labsl_civil_time \
	-labsl_time_zone \
	-labsl_strings \
	-labsl_strings_internal \
	-latomic \
	-lrt \
	-labsl_base \
	-labsl_spinlock_wait \
	-labsl_int128 \
	-labsl_throw_delegate \
	-labsl_raw_logging_internal \
	-labsl_log_severity \
	-lusb-1.0
```

- Build
```bash
cd libedgetpu/makefile_build
TFROOT=/home/${USER}/Repos/tensorflow make -j2 libedgetpu
```

- Install
```bash
sudo cp /home/${USER}/Repos/libedgetpu/out/direct/k8/libedgetpu.so.1.0 /usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0
```