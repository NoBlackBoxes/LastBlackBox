#!/bin/bash
set -eu

# Set Roots
NBB_ROOT="/home/kampff/NoBlackBoxes"
LBB_ROOT=$NBB_ROOT"/repos/LastBlackBox"

# Set CPU and Modules folder
CPU=$LBB_ROOT"/boxes/fpgas/hdl/verilog/cpu"
MODULES=$CPU"/modules"

# Create out directory
mkdir -p bin

# Build CPU
iverilog -o bin/verify_cpu $CPU/cpu.v $MODULES/datapath.v $MODULES/flopr.v $MODULES/adder.v $MODULES/mux2.v $MODULES/regfile.v $MODULES/generate_immediate.v $MODULES/alu.v $MODULES/mux4.v $MODULES/controller.v $MODULES/main_decoder.v $MODULES/alu_decoder.v verify_cpu_tb.v

# Reset CPU
CPU=$LBB_ROOT"/boxes/fpgas/hdl/verilog/cpu/verification/bin/verify_cpu"

# Set toolchain
RISCV_TOOLCHAIN=$NBB_ROOT"/tools/rv32i-toolchain/bin"

# Set include folder
INCLUDE_DIR="$LBB_ROOT/boxes/fpgas/hdl/verilog/cpu/verification/include"

# Set Assembly Flags
ASFLAGS+=" -O0"                     # Don't perform optimizations
ASFLAGS+=" -Wall"                   # Report all warnings
ASFLAGS+=" -march=rv32i"            # Just the core RV32I ISA
ASFLAGS+=" -nostartfiles"           # No extra startup code
ASFLAGS+=" -x assembler-with-cpp"   # Use C preprocesser with assembler
ASFLAGS+=" -I "$INCLUDE_DIR         # Inlcude
ASFLAGS+=" -nostdlib"
ASFLAGS+=" --specs=nosys.specs"
#ASFLAGS+=" -Wl,-Tlink.ld"

# Run Tests

# ADD
$RISCV_TOOLCHAIN/riscv32-unknown-elf-gcc $ASFLAGS -o bin/add.o tests/add.S
$RISCV_TOOLCHAIN/riscv32-unknown-elf-objcopy -O binary -j .text bin/add.o bin/add.bin
hexdump bin/add.bin > bin/add.dump
python utilities/dump2machine.py bin/add.dump
cp bin/add.txt bin/imem.txt
echo "Test: add"
vvp $CPU

# ADDI
$RISCV_TOOLCHAIN/riscv32-unknown-elf-gcc $ASFLAGS -o bin/addi.o tests/addi.S
$RISCV_TOOLCHAIN/riscv32-unknown-elf-objcopy -O binary -j .text bin/addi.o bin/addi.bin
hexdump bin/addi.bin > bin/addi.dump
python utilities/dump2machine.py bin/addi.dump
cp bin/addi.txt bin/imem.txt
echo "Test: addi"
vvp $CPU

# AND
$RISCV_TOOLCHAIN/riscv32-unknown-elf-gcc $ASFLAGS -o bin/and.o tests/and.S
$RISCV_TOOLCHAIN/riscv32-unknown-elf-objcopy -O binary -j .text bin/and.o bin/and.bin
hexdump bin/and.bin > bin/and.dump
python utilities/dump2machine.py bin/and.dump
cp bin/and.txt bin/imem.txt
echo "Test: and"
vvp $CPU

# ANDI
$RISCV_TOOLCHAIN/riscv32-unknown-elf-gcc $ASFLAGS -o bin/andi.o tests/andi.S
$RISCV_TOOLCHAIN/riscv32-unknown-elf-objcopy -O binary -j .text bin/andi.o bin/andi.bin
hexdump bin/andi.bin > bin/andi.dump
python utilities/dump2machine.py bin/andi.dump
cp bin/andi.txt bin/imem.txt
echo "Test: andi"
vvp $CPU

## AUIPC
#$RISCV_TOOLCHAIN/riscv32-unknown-elf-gcc $ASFLAGS -o bin/auipc.o tests/auipc.S
#$RISCV_TOOLCHAIN/riscv32-unknown-elf-objcopy -O binary -j .text bin/auipc.o bin/auipc.bin
#hexdump bin/auipc.bin > bin/auipc.dump
#python utilities/dump2machine.py bin/auipc.dump
#cp bin/auipc.txt bin/imem.txt
#echo "Test: auipc"
#vvp $CPU
