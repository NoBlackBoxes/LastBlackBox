#!/bin/bash
set -eu

# Set Roots
NBB_ROOT="/home/kampff/NoBlackBoxes"
LBB_ROOT=$NBB_ROOT"/repos/LastBlackBox"

# Set NBBPU, Modules, and Tests folders
NBBPU=$LBB_ROOT"/boxes/computers/NBBPU"
MODULES=$NBBPU"/verilog/modules"

# Create out directory
mkdir -p bin
mkdir -p bin/bin

# Copy verilog module(s)
cp $NBBPU/verilog/nbbpu.v bin/.
cp $MODULES/* bin/.

# Copy constraints file
cp upduino.pcf bin/.

# Copy memory contents
cp rom.txt bin/bin/.
cp ram.txt bin/bin/.

# Verify Verilog
apio verify --project-dir=bin --board upduino3 --verbose

# Synthesize
#apio build --project-dir=bin --board upduino3 --verbose
cd bin
nextpnr-ice40 --up5k --package sg48 --json hardware.json --asc hardware.asc --pcf upduino.pcf --pcf-allow-unconstrained

# FIN