#!/bin/bash
set -eu

# Set Roots
NBB_ROOT="/home/kampff/NoBlackBoxes"
LBB_ROOT=$NBB_ROOT"/repos/LastBlackBox"

# Set NBBPU and Modules folders
NBBPU=$LBB_ROOT"/boxes/computers/NBBPU"
MODULES=$NBBPU"/verilog/modules"

# Create out directory
rm -rf bin
mkdir -p bin
mkdir -p bin/bin

# Copy verilog module(s)
cp $NBBPU/verilog/nbbsoc.v bin/.
cp $NBBPU/verilog/nbbpu.v bin/.
cp $MODULES/* bin/.

# Copy constraints file
cp nbbsoc.pcf bin/.

# Copy memory contents
cp rom.txt bin/bin/.
cp ram.txt bin/bin/.

# Verify Verilog
apio verify --project-dir=bin --board upduino3 --verbose

# Synthesize
apio build --project-dir=bin --board upduino3 --verbose
#cd bin
#nextpnr-ice40 --up5k --package sg48 --json hardware.json --asc hardware.asc --pcf nbbsoc.pcf --pcf-allow-unconstrained

# Upload
#apio upload --project-dir=bin --board upduino3 --verbose
# FIN