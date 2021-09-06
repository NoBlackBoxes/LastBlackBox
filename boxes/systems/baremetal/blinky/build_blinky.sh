#!/bin/bash
set -eu

# Set Root
NBB_ROOT="/home/kampff/NoBlackBoxes"
LBB_ROOT=$NBB_ROOT"/repos/LastBlackBox"

# Set ARM toolchain
ARM_TOOLCHAIN=$NBB_ROOT"/tools/arm-toolchain/bin/arm-none-eabi-"

# Create output directory
mkdir -p bin

# Specify C Flags
CFLAGS="-g"
CFLAGS+=" -nostartfiles"
CFLAGS+=" -mfloat-abi=hard"
CFLAGS+=" -O0"
CFLAGS+=" -DRPI4"
CFLAGS+=" -mfpu=crypto-neon-fp-armv8"
CFLAGS+=" -march=armv8-a+crc"
CFLAGS+=" -mcpu=cortex-a72"

# Compile
${ARM_TOOLCHAIN}gcc ${CFLAGS} -I ../include blinky.c -o bin/kernel.elf

# FIN