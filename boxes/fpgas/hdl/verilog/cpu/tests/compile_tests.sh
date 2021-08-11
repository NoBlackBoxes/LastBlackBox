#!/bin/bash
set -eu

# Set toolchain
RISCV_TOOLCHAIN="/home/kampff/NoBlackBoxes/tools/rv32i-toolchain/bin"

# Create out directory
mkdir -p bin

# Compile
$RISCV_TOOLCHAIN/riscv32-unknown-elf-gcc -S -o bin/simple.s simple.c
