#!/bin/bash
set -eu

# Set Roots
NBB_ROOT="/home/kampff/NoBlackBoxes"
LBB_ROOT="/home/kampff/NoBlackBoxes/repos/LastBlackBox"

# Set CPU
CPU=$LBB_ROOT"/boxes/fpgas/hdl/verilog/cpu/verification/bin/verify_cpu"

# Set toolchain
RISCV_TOOLCHAIN=$NBB_ROOT"/tools/rv32i-toolchain/bin"

# Set include folder
INCLUDE_DIR="$NBB_ROOT/repos/LastBlackBox/boxes/fpgas/hdl/verilog/cpu/verification/include"

# Create out directoryss
mkdir -p bin

# Assemble
ASFLAGS+=" -O0"                     # Don't perform optimizations
ASFLAGS+=" -Wall"                   # Report all warnings
ASFLAGS+=" -march=rv32i"            # Just the core RV32I ISA
ASFLAGS+=" -nostartfiles"           # No extra startup code
ASFLAGS+=" -x assembler-with-cpp"   # Use C preprocesser with assembler
ASFLAGS+=" -I "$INCLUDE_DIR         # Inlcude
ASFLAGS+=" -nostdlib"
ASFLAGS+=" --specs=nosys.specs"
#ASFLAGS+=" -Wl,-Tlink.ld"
$RISCV_TOOLCHAIN/riscv32-unknown-elf-gcc $ASFLAGS -o bin/addi.o addi.S

# Extract binary object
$RISCV_TOOLCHAIN/riscv32-unknown-elf-objcopy -O binary -j .text bin/addi.o bin/addi.bin

# Parse binary into HEX tmachone code file for Verilog
hexdump bin/addi.bin > bin/addi.dump
python ../../utilities/dump2machine.py bin/addi.dump
cp bin/addi.txt bin/imem.txt

# Simulate
vvp $CPU
