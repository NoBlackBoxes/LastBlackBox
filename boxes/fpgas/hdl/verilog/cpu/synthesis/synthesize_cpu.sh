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
mkdir -p bin/bin

# Copy verilog module(s)
cp $CPU/cpu.v bin/.
cp $MODULES/* bin/.

# Copy constraints file
cp upduino.pcf bin/.

# Copy memory contents
cp rom.txt bin/bin/.
cp ram.txt bin/bin/.

# Verify Verilog
apio verify --project-dir=bin --board upduino3 --verbose

# Synthesize
apio build --project-dir=bin --board upduino3 --verbose

# FIN