# Computers : Pi Pico
Instructions for using the Raspberry Pi Pico series microcontrollers

## Prerequisites
```bash
# Arch Linux
sudo pacman -S cmake base-devel arm-none-eabi-gcc arm-none-eabi-newlib arm-none-eabi-binutils arm-none-eabi-gdb

# Ubuntu/Debian
sudo apt install cmake python3 build-essential gcc-arm-none-eabi libnewlib-arm-none-eabi libstdc++-arm-none-eabi-newlib
```
- Add UDEV rules for user device access
```bash
# Ubuntu/Debian
sudo cp /home/${USER}/NoBlackBoxes/LastBlackBox/boxes/computers/pi-pico/_resources/udev/99-pico_debian.rules/etc/udev/rules.d/.

# Arch
sudo cp /home/${USER}/NoBlackBoxes/LastBlackBox/boxes/computers/pi-pico/_resources/udev/99-pico_arch.rules /etc/udev/rules.d/.

# Reload UDEV rule
sudo udevadm control --reload-rules
sudo udevadm trigger
```

## Setting up the Pico SDK

```bash
# Create a temporary folder to store the SDK
cd ~/NoBlackBoxes/LastBlackBox/_tmp
mkdir pico
cd pico

# Clone the Pico SDK repo (with submodules)
git clone --recurse-submodules https://github.com/raspberrypi/pico-sdk
```

## Compile the blink example

```bash
cd ~/NoBlackBoxes/LastBlackBox/boxes/computers/pi-pico/blink
mkdir build
cd build
cmake ..
## For Pico W
## cmake -DPICO_BOARD=pico_w ..
make
```

## Run the blink example
- Reset the Pico with the "bootsel" button pressed
- Copy the blink.ufd file from the build folder to the Pico

## Uploading code using picotool
- Install prerequisites
```bash
# Arch Linux
sudo pacman -S pkgconfig libusb
 
# Ubuntu/Debian
sudo apt install build-essential pkg-config libusb-1.0-0-dev cmake
```

- Clone the picotool repo into the _tmp/pico folder
```bash
cd ~/NoBlackBoxes/LastBlackBox/_tmp/pico
git clone https://github.com/raspberrypi/picotool
cd picotool
```
- Build picotool
```bash
mkdir build
cd build
cmake -DPICO_SDK_PATH="/home/${USER}/NoBlackBoxes/LastBlackBox/_tmp/pico/pico-sdk" ..
make
```
- Install picotool for use by system
```bash
sudo make install
```

You can now upload code with the following command (but only when the Pico is in "bootsel" mode):

```bash
picotool load <name of program>.uf2 -f
```

## Monitor USB port
```bash
minicom -D /dev/ttyACM0 -b 115200
```
