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
$RISCV_TOOLCHAIN/riscv32-unknown-elf-gcc $ASFLAGS -o bin/add.o add.s

# Extract binary object
$RISCV_TOOLCHAIN/riscv32-unknown-elf-objcopy -O binary -j .text bin/add.o bin/add.bin

# Parse binary into HEX tmachone code file for Verilog
hexdump bin/add.bin > bin/add.dump
python ../../utilities/dump2machine.py bin/add.dump
cp bin/add.txt bin/imem.txt

# Simulate
vvp $NBB_ROOT"/repos/LastBlackBox/boxes/fpgas/hdl/verilog/cpu/testbenches/bin/cpu"
