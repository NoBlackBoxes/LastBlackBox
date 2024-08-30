# systems : NBBOS

A "no black box" operating system (for the Rapsberry Pi 4b)

## Credit

This code is based on a number of spectacular tutorials and repos. Listed below.

- [isometime's Pi 4b Tutorial](https://github.com/isometimes/rpi4-osdev/tree/master)
- [Leon de Boer's repo](https://github.com/LdB-ECM/Raspberry-Pi/tree/master) and [great forum post](https://forums.raspberrypi.com/viewtopic.php?t=213964#p1317689)
- [Rythym16's Pi 4b MMU example](https://github.com/rhythm16/rpi4-bare-metal)
- [Low Level Devel's videos](https://www.youtube.com/channel/UCRWXAQsN5S3FPDHY4Ttq1Xg/videos)

Also, my MMU and Cache activation assembly is a very light mod of Arm's [Baremetal Bootcode for Armv8 Guide](http://classweb.ece.umd.edu/enee447.S2021/baremetal_boot_code_for_ARMv8_A_processors.pdf)

## Prerequisites

- Download cross-compile toolchain: https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads
- Extract it to a temporary folder within the LBB repo (boxes/systems/NBBOS) tree

```bash
mkdir _tmp
cd _tmp
tar -xhf gcc-arm*
```

- Clone and build usbboot (to write to eMMC on CM4)
```bash
git clone --depth=1 https://github.com/raspberrypi/usbboot
cd usbboot
make
```

- Use rpiboot to mount boot partition and copy a new kernel image

## Debugging

- UART connection as follows

```bash
minicom -D /dev/ttyUSB0 -b 115200
```
