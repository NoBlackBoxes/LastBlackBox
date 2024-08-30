# Audio : i2s : Driver for NB3 Ears and Mouth

***Important***: This driver installation only applies to *Revision 2* NB3_Mouth boards.

### Change OS Configuration 

The Raspberry Pi does not enable i2s by default. You can enable it by opening the file called "config.txt" in the /"boot" folder of your Raspberry Pi and changing a single line.

We also need to enable a driver for the NB£ Mouth board. The board uses an I2S codec + amplifier chip called the MAX98357A and driver is included in Linux. We only need to tell the OS that we want to use this driver by adding a "sevice tree overlay". This will require adding one more line to the configuration text file.

```bash
sudo nano /boot/firmware/config.txt
```

Then use the text editor to change and add the following lines:

```txt
#dtparam=i2s=on
```
...becomes

```txt
dtparam=i2s=on
dtoverlay=max98357a,sdmode-pin=16
```

### Install Kernal Headers

```bash
# Download the Linux kernel headers (for RPi's current kernel version) - after running update/upgrade
sudo apt-get install raspberrypi-kernel-headers
```

### Driver module

The NB3 Ear and NB3 mouth boards need a special driver that is not included by default in the Raspberry Pi Linux kernel. Thus you will have to build (compile and link) the driver as a "kernel module" and then install it in your system. Here are the steps.

```bash
# Clone the LastBlackBox repo (if you have not done so already!)
cd ~
mkdir NoBlackBoxes
cd NoBlackBoxes 
git clone --depth 1 --recurse-submodules https://github.com/NoBlackBoxes/LastBlackBox

# Navigate to the i2s/driver folder
cd ~/NoBlackBoxes/LastBlackBox/boxes/audio/i2s/driver

# Run the Makefile
make all
# - this will build the kernel module (*.ko) file from the nb3-audio-module.c source file.

# Install the module
sudo make install

# Insert the module into the kernel
sudo insmod nb3-audio-module.ko
```

You will have to insert the module each time you reboot (run the insmod command above). If you want to have it load automatically then you need to add its name to this file: "/etc/modules".

```bash
sudo nano /etc/modules
```

and add...

```txt
nb3-audio-module
```

***After this, power down your NB3 and complete the hardware installation.*** Follow the instructions [here](../README.md).

## Testing

If the driver (module) is installed and loaded, then you should be able to record audio. Check that a device is available:

```bash
arecord -l
```

Should output something like this...

```txt
**** List of CAPTURE Hardware Devices ****
card 3: MAX98357A [MAX98357A], device 0: fe203000.i2s-HiFi HiFi-0 [fe203000.i2s-HiFi HiFi-0]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
```

To make a test recording.

```bash
arecord -D plughw:3 -c2 -r 44100 -f S32 -t wav -V stereo -v file_stereo.wav
```

To test playback.

```bash
aplay -D plughw:3 -c2 -r 44100 -f S32 -t wav -V stereo -v file_stereo.wav
```

***NOTE***: The card number might change each reboot. If you notice a problem, then check the card number with "arecord -l or aplay -l"