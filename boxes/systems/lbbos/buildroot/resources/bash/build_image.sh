#!/bin/bash
set -e

# Find number of cores available for build jobs
NJOB=`sed -n "N;/processor/p" /proc/cpuinfo|wc -l`

# Check for LBBROOT environment variable
if test -z "${LBBROOT}" 
then
      echo "\$LBBROOT is not set (exiting)"
      exit 0
fi

# Set environment variables
LBBREPO=${LBBROOT}"/repo"
LBBTOOLS=${LBBROOT}"/tools"
LBBBUILDROOT=${LBBTOOLS}"/buildroot"
LBBBUILDROOT_EXTERNAL=${LBBREPO}"/boxes/systems/LBBOS/resources/buildroot"

# Delete existing images
sudo rm -rf ${LBBTOOLS}/images
mkdir -p ${LBBTOOLS}/images

# Set genimage environment variables
GENIMAGE_CFG="${LBBBUILDROOT_EXTERNAL}/board/LastBlackBox/LBB/genimage-LBB.cfg"
GENIMAGE_TMP="${LBBTOOLS}/images/genimage.tmp"
GENIMAGE_ROOT="$(mktemp -d)" # Empty root path

# Assemble Image using genimage script
# boot/
# rootfs/

# Copy kernel image and DTB to boot folder
cp ${LBBTOOLS}/kernel/arch/arm64/boot/Image ${LBBTOOLS}/images
cp ${LBBTOOLS}/kernel/arch/arm64/boot/dts/broadcom/bcm2711-rpi-4-b.dtb ${LBBTOOLS}/images

# Copy firmware folder
cp -r ${LBBBUILDROOT}/output/LBB/images/LBB-firmware ${LBBTOOLS}/images

# Copy rootfs (ext2 and symlink)
cp ${LBBBUILDROOT}/output/LBB/images/rootfs.ext2 ${LBBTOOLS}/images/rootfs.ext2
cp ${LBBBUILDROOT}/output/LBB/images/rootfs.ext4 ${LBBTOOLS}/images/rootfs.ext4

# Mount rootfs
mkdir -p /tmp/rootfs
sudo mount ${LBBTOOLS}/images/rootfs.ext4 /tmp/rootfs

# Add kernel modules
sudo mkdir -p /tmp/rootfs/lib/modules
sudo cp -af ${LBBTOOLS}/kernel/modules/lib/modules/* /tmp/rootfs/lib/modules

# Overlay FS (also replaces /etc/sudoers)...should be in fakeroot(?)
#sudo rsync -a --exclude='.git' ${LBBREPO}/resources/buildroot/board/LastBlackBox/LBB/overlay/* /tmp/rootfs

# Unmount rootfs
sudo umount /tmp/rootfs
rm -rf /tmp/rootfs

# Change to "images" folder
cd ${LBBTOOLS}/images

# Generate timage
${LBBBUILDROOT}/output/LBB/host/bin/genimage \
	--rootpath "${GENIMAGE_ROOT}"   \
	--tmppath "${GENIMAGE_TMP}"    \
	--inputpath "${LBBTOOLS}/images"  \
	--outputpath "${LBBTOOLS}/images" \
	--config "${GENIMAGE_CFG}"

echo "FIN"
exit 0
#FIN