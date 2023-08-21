#!/bin/bash
set -eu

# Set Roots
NBB_ROOT="/home/kampff/NoBlackBoxes"
LBB_ROOT=$NBB_ROOT"/repos/LastBlackBox"

# Set NBBPU, Modules, and Tests folders
NBBPU=$LBB_ROOT"/boxes/computers/NBBPU"
MODULES=$NBBPU"/verilog/modules"
TESTS=$NBBPU"/verilog/verification/tests/*.as"
ASSEMBLER=$NBBPU"/assembler/assemble.py"

# Create (new) output directory
rm -rf bin
mkdir -p bin

# Build CPU
iverilog -o bin/verify_nbbpu \
    $NBBPU/verilog/nbbpu.v \
    $MODULES/controller.v \
    $MODULES/regfile.v \
    $MODULES/alu.v \
    $MODULES/rom.v \
    $MODULES/ram.v \
    $MODULES/flopenr.v \
    $MODULES/mux2.v \
    verify_nbbpu_tb.v

# Create empty machine data file for initializing RAM
python $NBBPU/verilog/utilities/empty_ram.py "bin/ram.txt"

# Run Tests
echo "------------------"
echo "NBBPU Verification"
echo "------------------"
for t in $TESTS
do
    FILENAME=$(basename -- "$t")
    TESTNAME="${FILENAME%.*}"
    echo "Testing: $TESTNAME ($FILENAME)"

    # Assemble test
    python $ASSEMBLER tests/$FILENAME bin/$TESTNAME.rom

    # Copy
    cp bin/$TESTNAME.rom bin/rom.txt

    # Simulate
    vvp bin/verify_nbbpu

    echo "-----"

done
