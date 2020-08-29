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

# Setup buildroot external tree
cd ${LBBBUILDROOT}
mkdir -p output/LBB
make O=output/LBB BR2_EXTERNAL=${LBBBUILDROOT_EXTERNAL} LBB_defconfig
cd output/LBB

# Build toolchain
make toolchain O=output/LBB -j${NJOB}

echo "FIN"
exit 0
#FIN