# systems : lbbos : buildroot

Instructions for building a 64-bit OS with Buildroot for a Raspberry Pi 4b based on the Broadcom BCM2711 system-on-chip

## Download Tools

- **Buildroot** (version 2020.05.x) [buildroot](https://github.com/buildroot/buildroot/tree/2020.05.x)
  - Extract to LBBROOT/tools/OS/buildroot
- **Kernel** (version 5.7) [kernel](https://github.com/raspberrypi/linux/tree/rpi-5.7.y)
  - Extract to LBBROOT/tools/OS/kernel

## Build LastBlackBox OS

- Set LBBROOT environment variable

```bash
export LBBROOT="path to your LastBlackBox root directory"
echo $LBBROOT
```

- Build toolchain (LBBROOT/boxes/systems/lbbos/buildroot/resources/bash/build_toolchain.sh)
- Build kernel (LBBROOT/boxes/systems/lbbos/buildroot/resources/bash/build_kernel.sh)
- Build root filesystem (LBBROOT/boxes/systems/lbbos/buildroot/resources/bash/build_rootfs.sh)
- Build image (LBBROOT/boxes/systems/lbbos/buildroot/resources/bash/build_image.sh)

## Notes

- ***REMEMBER to add student to sudoers file "/etc/sudoers": at end - "student  ALL=(ALL) NOPASSWD: ALL"***
- ***REMEMBER copy drm.h thing*** in the toolchain sysroot
- Load modules

```bash
sudo modprobe bcm2835-v4l2 max_video_width=2592 max_video_height=1944
sudo modprobe vc4
sudo modprobe v3d
```
