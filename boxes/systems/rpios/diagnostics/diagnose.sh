#!/bin/bash

# Check root privileges
[ "$(whoami)" == "root" ] || { echo "Must be run as sudo!"; exit 1; }

# Define color variables
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[1;36m'
CLEAR='\033[0m'

# Clear Screen
clear
sync
echo -e "Raspberry ${RED}Pi${CLEAR} Diagnostics"
echo -e "${YELLOW}------------------------${CLEAR}"

## Show current hardware settings
echo -e "${CYAN}"
vcgencmd measure_temp
vcgencmd get_config int | grep arm_freq
vcgencmd get_config int | grep core_freq
vcgencmd get_config int | grep sdram_freq
vcgencmd get_config int | grep gpu_freq
printf "sd_clock="
grep "actual clock" /sys/kernel/debug/mmc0/ios 2>/dev/null | awk '{printf("%0.3f MHz\n", $3/1000000)}'
echo -e "${CLEAR}"

# Test Internet speeds
echo -e "${YELLOW}Running Internet Speed test...${CLEAR}"
speedtest-cli --simple
echo -e "${CLEAR}"

# Test CPU performance
echo -e "${YELLOW}Running CPU test ($(nproc) cores)...${CLEAR}"
sysbench cpu --threads="$(nproc)" run | grep 'total time:\|min:\|avg:\|max:' | tr -s [:space:]
echo -e "${RED}"
vcgencmd measure_temp
echo -e "${CLEAR}"

# Test Threading performance
echo -e "${YELLOW}Running THREADS test ($(nproc) cores)...${CLEAR}"
sysbench threads --num-threads=4 --validate=on --thread-yields=4000 --thread-locks=6 run | grep 'total time:\|min:\|avg:\|max:' | tr -s [:space:]
echo -e "${RED}"
vcgencmd measure_temp
echo -e "${CLEAR}"

# Test Memory (RAM) performance
echo -e "${YELLOW}Running MEMORY test...${CLEAR}"
sysbench memory --num-threads=4 --validate=on --memory-block-size=1K --memory-total-size=3G --memory-access-mode=seq run | grep 'Operations\|transferred\|total time:\|min:\|avg:\|max:' | tr -s [:space:]
echo -e "${RED}"
vcgencmd measure_temp
echo -e "${CLEAR}"

# Test SD Card performance
echo -e "${YELLOW}Running SD Card (hdparm) test...${CLEAR}"
hdparm -t /dev/mmcblk0 | grep Timing
echo -e "${RED}"
vcgencmd measure_temp
echo -e "${CLEAR}"

# Test SD Care WRITE performance
echo -e "${YELLOW}Running DD WRITE test...${CLEAR}"
rm -f ~/test.tmp && sync && dd if=/dev/zero of=~/test.tmp bs=1M count=512 conv=fsync 2>&1 | grep -v records
echo -e "${RED}"
vcgencmd measure_temp
echo -e "${CLEAR}"

# Test SD Care READ performance
echo -e "${YELLOW}Running DD READ test...${CLEAR}"
echo -e 3 > /proc/sys/vm/drop_caches && sync && dd if=~/test.tmp of=/dev/null bs=1M 2>&1 | grep -v records
echo -e "${RED}"
vcgencmd measure_temp
rm -f ~/test.tmp
echo -e "${CLEAR}"

echo -e "${YELLOW}------------------------${CLEAR}"
echo -e "\n"

# FIN