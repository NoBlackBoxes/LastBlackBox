#!/bin/bash
set -eu

# Set toolchain
RISCV_TOOLCHAIN="/home/kampff/NoBlackBoxes/tools/rv32i-toolchain/bin"

# Create out directory
mkdir -p bin

# Compile
COMPILE_FLAGS=
$RISCV_TOOLCHAIN/riscv32-unknown-elf-gcc -S -o bin/simple.s simple.c

# Assemble
ASFLAGS+=-O0                  # Don't perform optimizations
ASFLAGS+=-Wall                # Report all warnings
ASFLAGS+=-march=rv32i         # Just the core RV32I ISA
ASFLAGS+=-nostartfiles        # No extra startup code
ASFLAGS+=-nostdlib
ASFLAGS+=--specs=nosys.specs
ASFLAGS+=-Wl,-Tlink.ld

$RISCV_TOOLCHAIN/riscv32-unknown-elf-as -o bin/simple.o bin/simple.s

# Extract binary
$RISCV_TOOLCHAIN/riscv32-unknown-elf-objdump -s bin/simple.o

#$RISCV_TOOLCHAIN/riscv32-unknown-elf-objcopy -O binary -j .text bin/simple.out simple.bin


#hexdump bin/simple.out