# Computers : Pi Pico
Instructions for using the Raspberry Pi Pico series microcontrollers

## Prerequsities
```bash
# Arch Linux
sudo pacman -S cmake base-devel arm-none-eabi-gcc arm-none-eabi-newlib arm-none-eabi-binutils arm-none-eabi-gdb

# Ubuntu/Debian
sudo apt install cmake python3 build-essential gcc-arm-none-eabi libnewlib-arm-none-eabi libstdc++-arm-none-eabi-newlib
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