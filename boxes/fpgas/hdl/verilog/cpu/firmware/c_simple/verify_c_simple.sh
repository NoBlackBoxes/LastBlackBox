#!/bin/bash
set -eu

# Set Root
NBB_ROOT="/home/kampff/NoBlackBoxes"

# Set toolchain
RISCV_TOOLCHAIN=$NBB_ROOT"/tools/rv32i-toolchain/bin"

# Set Tests include folder
TESTS_INCLUDES="$NBB_ROOT/repos/LastBlackBox/boxes/fpgas/hdl/verilog/cpu/verification/tests"

# Create out directoryss
mkdir -p bin

# ELF
CFLAGS+=" -O0"                     # Don't perform optimizations
CFLAGS+=" -Wall"                   # Report all warnings
CFLAGS+=" -march=rv32i"            # Just the core RV32I ISA
CFLAGS+=" -nostartfiles"           # No extra startup code
CFLAGS+=" -nostdlib"
CFLAGS+=" --specs=nosys.specs"
$RISCV_TOOLCHAIN/riscv32-unknown-elf-gcc $CFLAGS -o bin/c_simple.out c_simple.c
$RISCV_TOOLCHAIN/riscv32-unknown-elf-objdump -s bin/c_simple.out > bin/c_simple_elf.dump

# Compile
CFLAGS+=" -O0"                     # Don't perform optimizations
CFLAGS+=" -Wall"                   # Report all warnings
CFLAGS+=" -march=rv32i"            # Just the core RV32I ISA
CFLAGS+=" -nostartfiles"           # No extra startup code
CFLAGS+=" -x assembler-with-cpp"   # Use C preprocesser with assembler
CFLAGS+=" -I "$TESTS_INCLUDES      # Inlcude
CFLAGS+=" -nostdlib"
CFLAGS+=" --specs=nosys.specs"
$RISCV_TOOLCHAIN/riscv32-unknown-elf-gcc -S -o bin/c_simple.s c_simple.c

# Assemble
ASFLAGS+=" -O0"                     # Don't perform optimizations
ASFLAGS+=" -Wall"                   # Report all warnings
ASFLAGS+=" -march=rv32i"            # Just the core RV32I ISA
ASFLAGS+=" -nostartfiles"           # No extra startup code
ASFLAGS+=" -x assembler-with-cpp"   # Use C preprocesser with assembler
ASFLAGS+=" -I "$TESTS_INCLUDES      # Inlcude
ASFLAGS+=" -nostdlib"
ASFLAGS+=" --specs=nosys.specs"
#ASFLAGS+=" -Wl,-Tlink.ld"
$RISCV_TOOLCHAIN/riscv32-unknown-elf-gcc $ASFLAGS -o bin/c_simple.o bin/c_simple.s

# Extract binary object
$RISCV_TOOLCHAIN/riscv32-unknown-elf-objcopy -O binary -j .text bin/c_simple.o bin/c_simple.bin

# Parse binary into HEX machine code file for Verilog
hexdump bin/c_simple.bin > bin/c_simple.dump
python ../../utilities/dump2machine.py bin/c_simple.dump
cp bin/c_simple.txt bin/imem.txt

# Simulate
vvp $NBB_ROOT"/repos/LastBlackBox/boxes/fpgas/hdl/verilog/cpu/testbenches/bin/cpu"
