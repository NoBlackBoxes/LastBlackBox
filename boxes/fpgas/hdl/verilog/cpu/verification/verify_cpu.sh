#!/bin/bash
set -eu

# Set Root
LBB_ROOT="/home/kampff/NoBlackBoxes/repos/LastBlackBox"

# Set CPU and Modules folder
CPU=$LBB_ROOT"/boxes/fpgas/hdl/verilog/cpu"
MODULES=$CPU"/modules"

# Create out directory
mkdir -p bin

# Build CPU
iverilog -o bin/verify_cpu $CPU/cpu.v $MODULES/datapath.v $MODULES/flopr.v $MODULES/adder.v $MODULES/mux2.v $MODULES/regfile.v $MODULES/generate_immediate.v $MODULES/alu.v $MODULES/mux4.v $MODULES/controller.v $MODULES/main_decoder.v $MODULES/alu_decoder.v verify_cpu_tb.v
