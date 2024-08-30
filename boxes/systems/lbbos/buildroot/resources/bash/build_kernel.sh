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
LBBTOOLCHAIN=${LBBBUILDROOT}"/output/LBB/host/bin"
LBBSYSROOT="${LBBBUILDROOT}/output/LBB/host/aarch64-LBB-linux-gnu/sysroot"

# Set (Cross) Compiler Root
CCROOT="${LBBTOOLCHAIN}/aarch64-LBB-linux-gnu-"

# Enter kernel root
cd ${LBBTOOLS}/kernel

# Build kernel (and DTB with __symbols__)
#make mrproper
make ARCH=arm64 CROSS_COMPILE=${CCROOT} bcm2711_defconfig
make ARCH=arm64 CROSS_COMPILE=${CCROOT} DTC_FLAGS=-@ --jobs=${NJOB}

# Build and install kernel modules
MOD_INSTALL_PATH=${LBBTOOLS}/kernel/modules
rm -rf ${MOD_INSTALL_PATH}
mkdir -p ${MOD_INSTALL_PATH}
make ARCH=arm64 CROSS_COMPILE=${CCROOT} bcm2711_defconfig
make ARCH=arm64 CROSS_COMPILE=${CCROOT} INSTALL_MOD_PATH=${MOD_INSTALL_PATH} modules -j$NJOB
make ARCH=arm64 CROSS_COMPILE=${CCROOT} INSTALL_MOD_PATH=${MOD_INSTALL_PATH} modules_install

echo "FIN"
exit 0
#FIN