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
iverilog -o bin/nbbpu \
    $MODULES/controller.v \
    $MODULES/datapath.v \
    $MODULES/regfile.v \
    $MODULES/alu.v \
    $MODULES/rom.v \
    $MODULES/ram.v \
    $MODULES/flopr.v \
    $MODULES/mux2.v \
    nbbpu.v nbbpu_tb.v

# Simulate
vvp bin/nbbpu

# Visualize
gtkwave bin/nbbpu_tb.vcd