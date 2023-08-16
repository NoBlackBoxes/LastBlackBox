#!/bin/bash
set -eu

# Set Root
LBB_ROOT="/home/kampff/NoBlackBoxes/repos/LastBlackBox"

# Set NBBPU and Modules folder
NBBPU=$LBB_ROOT"/boxes/computers/NBBPU/verilog"
MODULES=$NBBPU"/modules"

# Create out directory
mkdir -p bin

# Build
iverilog -o bin/ram \
    $MODULES/ram.v \
    ram_tb.v

# Simulate
vvp bin/ram

# Visualize
gtkwave bin/ram_tb.vcd