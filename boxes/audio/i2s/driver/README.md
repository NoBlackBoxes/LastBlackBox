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

### Install Kernel Headers

```bash
# Download the Linux kernel headers (for RPi's current kernel version) - after running update/upgrade
sudo apt-get install raspberrypi-kernel-headers
```

### Driver module

The NB3 Ear and NB3 mouth boards need a special driver that is not included by default in the Raspberry Pi Linux kernel. Thus you will have to build (compile and link) the driver as a "kernel module" and then install it in your system. Here are the steps.

```bash
# Navigate to the i2s/driver folder
cd ~/NoBlackBoxes/LastBlackBox/boxes/audio/i2s/driver

# Create a temporary folder to store the output of the build
mkdir _tmp

# Copy the required files to the temporary folder
cp Makefile _tmp/.
cp nb3-audio-module.c _tmp/.

# Navigate to the temporary folder
cd _tmp

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
