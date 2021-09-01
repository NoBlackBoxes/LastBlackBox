#!/bin/bash
set -eu

# Set Roots
NBB_ROOT="/home/kampff/NoBlackBoxes"
LBB_ROOT=$NBB_ROOT"/repos/LastBlackBox"

# Set CPU and Modules folder
CPU=$LBB_ROOT"/boxes/fpgas/hdl/verilo/cpu"
MODULES=$CPU"/modules"

# Create out directory
mkdir -p bin

# Verify Verilog
apio verify --project-dir=cpu --board upduino3 --verbose

# Synthesize
apio build --project-dir=cpu --board upduino3 --verbose


#iverilog -o bin/synthesize_cpu $CPU/cpu.v $MODULES/datapath.v $MODULES/flopr.v $MODULES/adder.v $MODULES/mux2.v $MODULES/mux3.v $MODULES/regfile.v $MODULES/generate_immediate.v $MODULES/alu.v $MODULES/mux8.v $MODULES/select_read.v $MODULES/rom.v $MODULES/ram.v $MODULES/controller.v $MODULES/main_decoder.v $MODULES/alu_decoder.v verify_cpu_tb.v

#