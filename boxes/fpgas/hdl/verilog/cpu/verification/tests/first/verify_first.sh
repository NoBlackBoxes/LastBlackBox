#!/bin/bash
set -eu

# Set toolchain
RISCV_TOOLCHAIN="/home/kampff/NoBlackBoxes/tools/rv32i-toolchain/bin"

# Create out directory
mkdir -p bin

# Assemble
ASFLAGS+=-O0                  # Don't perform optimizations
ASFLAGS+=-Wall                # Report all warnings
ASFLAGS+=-march=rv32i         # Just the core RV32I ISA
ASFLAGS+=-nostartfiles        # No extra startup code
ASFLAGS+=-nostdlib
ASFLAGS+=--specs=nosys.specs
ASFLAGS+=-Wl,-Tlink.ld
$RISCV_TOOLCHAIN/riscv32-unknown-elf-as -o bin/first.o first.s

# Extract binary object
$RISCV_TOOLCHAIN/riscv32-unknown-elf-objcopy -O binary -j .text bin/first.o bin/first.bin

# Parse binary into HEX text file for Verilog
hexdump bin/first.bin > bin/first.dump
python ../../utilities/dump2machine.py bin/first.dump
hexdump bin/first.bin
