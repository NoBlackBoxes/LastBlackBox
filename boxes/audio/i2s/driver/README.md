# hearing : i2s : driver

## Installation (Software)

### Install Kernal Headers

```bash
# Download the Linux kernel headers (for RPi's current kernel version) - after running update/upgrade
sudo apt-get install raspberrypi-kernel-headers
```

### Change Configuration 

The Raspberry Pi does not enable i2s by default. You can enable it by opening the file called "config.txt" in the /"boot" folder of your Raspberry Pi and changing a single line.

```bash
sudo nano /boot/config.txt
```

Then use the text editor to change the following (just uncomment the line):

```txt
#dtparam=i2s=on
```
...becomes

```txt
dtparam=i2s=on
```

***After this, you must reboot your Raspberry Pi to continue.***

### Driver module

The NB3 Ear board needs a special driver that is not included by default in the Raspberry Pi Linux kernel. Thus you will have to build (compile and link) the driver as a "kernel module" and then install it in your system. Here are the steps.

```bash
# Clone the LastBlackBox repo (if you have not done so already)
git clone https://github.com/NoBlackBoxes/LastBlackBox

# Navigate to the i2s/driver folder
cd LastBlackBox/boxes/audio/i2s/driver

# Run the Makefile
make all
# - this will build the kernel module (*.ko) file from the nb3-ear-module.c source file.

# Install the module
sudo make install

# Insert the module into the kernel
sudo insmod nb3-ear-module.ko
```

You will have to insert the module each time you reboot (run the insmod command above). If you want to have it load automatically then you need to add its name to this file: "/etc/modules".

```bash
sudo nano /etc/modules
```

and add...

```txt
nb3-ear-module
```

***After this, it is useful to reboot your Raspberry Pi to continue.***

## Testing

If the driver (module) is installed and loaded, then you should be able to record audio. Check that a device is available:

```bash
arecord -l
```

Should output something like this...

```txt
**** List of CAPTURE Hardware Devices ****
card 1: NB3earcard [NB3_ear_card], device 0: simple-card_codec_link snd-soc-dummy-dai-0 [simple-card_codec_link snd-soc-dummy-dai-0]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
```

To make a test recording.

```bash
arecord -D plughw:1 -c2 -r 48000 -f S32_LE -t wav -V stereo -v file_stereo.wav
```

To test playback.

```bash
aplay -D plughw:1 -c2 -r 48000 -f S32_LE -t wav -V stereo -v file_stereo.wav
```