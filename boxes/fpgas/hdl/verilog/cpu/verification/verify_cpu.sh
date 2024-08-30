#!/bin/bash
set -eu

# Set Roots
NBB_ROOT="/home/${USER}/NoBlackBoxes"
LBB_ROOT=$NBB_ROOT"/LastBlackBox"

# Set CPU and Modules folder
CPU=$LBB_ROOT"/boxes/fpgas/hdl/verilog/cpu"
MODULES=$CPU"/modules"

# Create out directory
mkdir -p bin

# Build CPU
iverilog -o bin/verify_cpu $CPU/cpu.v $MODULES/datapath.v $MODULES/flopr.v $MODULES/adder.v $MODULES/mux2.v $MODULES/mux3.v $MODULES/regfile.v $MODULES/generate_immediate.v $MODULES/alu.v $MODULES/mux8.v $MODULES/select_read.v $MODULES/rom.v $MODULES/ram.v $MODULES/controller.v $MODULES/main_decoder.v $MODULES/alu_decoder.v verify_cpu_tb.v

# Create empty machine data file for initializing RAM
python ../utilities/empty_ram.py "bin/ram.txt"

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
ASFLAGS+=" -Wl,-Tinclude/link.ld"

# Run Tests
TESTS_FILES=$LBB_ROOT"/boxes/fpgas/hdl/verilog/cpu/verification/tests/*.S"
for tf in $TESTS_FILES
do
    FILENAME=$(basename -- "$tf")
    TESTNAME="${FILENAME%.*}"
    echo "Testing: $TESTNAME"

    $RISCV_TOOLCHAIN/riscv32-unknown-elf-gcc $ASFLAGS -o bin/$TESTNAME.o tests/$TESTNAME.S

    $RISCV_TOOLCHAIN/riscv32-unknown-elf-objcopy -O binary -j .text bin/$TESTNAME.o bin/$TESTNAME.code
    $RISCV_TOOLCHAIN/riscv32-unknown-elf-objcopy -O binary -j .data bin/$TESTNAME.o bin/$TESTNAME.data

    hexdump bin/$TESTNAME.code > bin/$TESTNAME.dump_code
    hexdump bin/$TESTNAME.data > bin/$TESTNAME.dump_data

    python ../utilities/dump2machine_code.py bin/$TESTNAME.dump_code
    python ../utilities/dump2machine_data.py bin/$TESTNAME.dump_data

    cp bin/$TESTNAME.text_code bin/rom.txt
    cp bin/$TESTNAME.text_data bin/ram.txt

    vvp $CPU
done
