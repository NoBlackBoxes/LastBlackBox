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

# format_size
# Purpose: Formats raw disk and memory sizes from kibibytes (KiB) to largest unit
# Parameters:
#          1. RAW - the raw memory size (RAM/Swap) in kibibytes
# Returns:
#          Formatted memory size in KiB, MiB, GiB, or TiB
function format_size {
	RAW=$1 # mem size in KiB
	RESULT=$RAW
	local DENOM=1
	local UNIT="KiB"

	# ensure the raw value is a number, otherwise return blank
	re='^[0-9]+$'
	if ! [[ $RAW =~ $re ]] ; then
		echo "" 
		return 0
	fi

	if [ "$RAW" -ge 1073741824 ]; then
		DENOM=1073741824
		UNIT="TiB"
	elif [ "$RAW" -ge 1048576 ]; then
		DENOM=1048576
		UNIT="GiB"
	elif [ "$RAW" -ge 1024 ]; then
		DENOM=1024
		UNIT="MiB"
	fi

	# divide the raw result to get the corresponding formatted result (based on determined unit)
	RESULT=$(awk -v a="$RESULT" -v b="$DENOM" 'BEGIN { print a / b }')
	# shorten the formatted result to two decimal places (i.e. x.x)
	RESULT=$(echo $RESULT | awk -F. '{ printf "%0.1f",$1"."substr($2,1,2) }')
	# concat formatted result value with units and return result
	RESULT="$RESULT $UNIT"
	echo $RESULT
}

# Report system information
echo -e 
echo -e "${GREEN}"System Information:"${CLEAR}"
echo -e "${YELLOW}-------------------${CLEAR}"
UPTIME=$(uptime | awk -F'( |,|:)+' '{d=h=m=0; if ($7=="min") m=$6; else {if ($7~/^day/) {d=$6;h=$8;m=$9} else {h=$6;m=$7}}} {print d+0,"days,",h+0,"hours,",m+0,"minutes"}')
echo -e "Uptime     : $UPTIME"
# check for local lscpu installs
command -v lscpu >/dev/null 2>&1 && LOCAL_LSCPU=true || unset LOCAL_LSCPU
if [[ $ARCH = *aarch64* || $ARCH = *arm* ]] && [[ ! -z $LOCAL_LSCPU ]]; then
	CPU_PROC=$(lscpu | grep "Model name" | sed 's/Model name: *//g')
else
	CPU_PROC=$(awk -F: '/model name/ {name=$2} END {print name}' /proc/cpuinfo | sed 's/^[ \t]*//;s/[ \t]*$//')
fi
echo -e "Processor  : $CPU_PROC"
if [[ $ARCH = *aarch64* || $ARCH = *arm* ]] && [[ ! -z $LOCAL_LSCPU ]]; then
	CPU_CORES=$(lscpu | grep "^[[:blank:]]*CPU(s):" | sed 's/CPU(s): *//g')
	CPU_FREQ=$(lscpu | grep "CPU max MHz" | sed 's/CPU max MHz: *//g')
	[[ -z "$CPU_FREQ" ]] && CPU_FREQ="???"
	CPU_FREQ="${CPU_FREQ} MHz"
else
	CPU_CORES=$(awk -F: '/model name/ {core++} END {print core}' /proc/cpuinfo)
	CPU_FREQ=$(awk -F: ' /cpu MHz/ {freq=$2} END {print freq " MHz"}' /proc/cpuinfo | sed 's/^[ \t]*//;s/[ \t]*$//')
fi
echo -e "CPU cores  : $CPU_CORES @ $CPU_FREQ"
CPU_AES=$(cat /proc/cpuinfo | grep aes)
[[ -z "$CPU_AES" ]] && CPU_AES="\xE2\x9D\x8C Disabled" || CPU_AES="\xE2\x9C\x94 Enabled"
echo -e "AES-NI     : $CPU_AES"
CPU_VIRT=$(cat /proc/cpuinfo | grep 'vmx\|svm')
[[ -z "$CPU_VIRT" ]] && CPU_VIRT="\xE2\x9D\x8C Disabled" || CPU_VIRT="\xE2\x9C\x94 Enabled"
echo -e "VM-x/AMD-V : $CPU_VIRT"
TOTAL_RAM_RAW=$(free | awk 'NR==2 {print $2}')
TOTAL_RAM=$(format_size $TOTAL_RAM_RAW)
echo -e "RAM        : $TOTAL_RAM"
TOTAL_SWAP_RAW=$(free | grep Swap | awk '{ print $2 }')
TOTAL_SWAP=$(format_size $TOTAL_SWAP_RAW)
echo -e "Swap       : $TOTAL_SWAP"
# total disk size is calculated by adding all partitions of the types listed below (after the -t flags)
TOTAL_DISK_RAW=$(df -t simfs -t ext2 -t ext3 -t ext4 -t btrfs -t xfs -t vfat -t ntfs -t swap --total 2>/dev/null | grep total | awk '{ print $2 }')
TOTAL_DISK=$(format_size $TOTAL_DISK_RAW)
echo -e "Disk       : $TOTAL_DISK"
DISTRO=$(grep 'PRETTY_NAME' /etc/os-release | cut -d '"' -f 2 )
echo -e "Distro     : $DISTRO"
KERNEL=$(uname -r)
echo -e "Kernel     : $KERNEL"
VIRT=$(systemd-detect-virt 2>/dev/null)
VIRT=${VIRT^^} || VIRT="UNKNOWN"
echo -e "VM Type    : $VIRT"
[[ -z "$IPV4_CHECK" ]] && ONLINE="\xE2\x9D\x8C Offline / " || ONLINE="\xE2\x9C\x94 Online / "
[[ -z "$IPV6_CHECK" ]] && ONLINE+="\xE2\x9D\x8C Offline" || ONLINE+="\xE2\x9C\x94 Online"
echo -e "IPv4/IPv6  : $ONLINE"

# Show current hardware settings
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